"""Learn separate classification and post-processing representations from matched image views."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from detection.training import seed_all
from lqy.codec_patch import labeled_data, tables_by_frequency
from lqy_resize_retrain import training_split
from native_texture_base import KERNELS, SEED, acquisition_scale, fit
from patch_evidence_texture import AnchorContextTexture, cache_training_views, grid_supervision
from self_supervised_texture_base import AUXILIARY_WEIGHT, texture_batch, training_assignments
from two_stage_texture_base import write_summary


MODEL = "degradation_anchor_texture"
TREATMENT_WEIGHT = .10
REPRESENTATION_WEIGHT = .10
KERNEL_COUNT = len(KERNELS) + 1


class DegradationAnchorTexture(AnchorContextTexture):
    def __init__(self):
        super().__init__()
        width = self.attention.in_features
        self.forensic_projection = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32), nn.GELU(),
                                                 nn.Linear(32, width, bias=False))
        nn.init.zeros_(self.forensic_projection[-1].weight)
        self.processing_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 32), nn.GELU(),
                                             nn.Linear(32, KERNEL_COUNT + 6))

    def forensic_features(self, patches):
        return patches + self.forensic_projection(patches)

    def classify_with_context(self, patches, overlap, valid, anchors):
        return super().classify_with_context(self.forensic_features(patches), overlap, valid, anchors)

    def training_outputs(self, patches, overlap, valid, anchors):
        forensic = self.forensic_features(patches)
        logits, anchor_logits = super().classify_with_context(forensic, overlap, valid, anchors)
        forensic_representation = self.pool_valid(forensic, anchors)
        processing_input = patches.sum(dim=1) / valid.sum(dim=1, keepdim=True)
        return logits, anchor_logits, forensic, forensic_representation, self.processing_head(processing_input)


def treatment_target(size, assignment):
    condition = assignment[0] % 4
    scale = acquisition_scale(size, assignment) if condition >= 2 else 1.0
    kernel = 1 + assignment[2] % len(KERNELS) if condition >= 2 else 0
    jpeg = condition % 2
    quality = ((66 + assignment[3] % 29) - 80) / 15 if jpeg else 0
    sampling = assignment[4] % 3 if jpeg else 0
    return (scale - 1) / .4, kernel, jpeg, quality, sampling


def sample_treatments(sample):
    with Image.open(sample.path) as image:
        size = image.size
    return tuple(treatment_target(size, assignment) for assignment in training_assignments(sample))


def training_views(samples, table, deadline):
    views = cache_training_views(samples, table, deadline, include_anchors=True)
    with ThreadPoolExecutor(max_workers=4) as workers:
        treatments = torch.tensor(list(workers.map(sample_treatments, samples)), dtype=torch.float32)
    return tuple((*view, target) for view, target in zip(views, treatments.unbind(dim=1), strict=True))


def processing_loss(prediction, target):
    scale, kernel, jpeg, quality, sampling = target.unbind(dim=1)
    scale_prediction, kernel_logits, jpeg_logits, quality_prediction, sampling_logits = prediction.float().split(
        (1, KERNEL_COUNT, 1, 1, 3), dim=1)
    unmasked = (F.smooth_l1_loss(scale_prediction[:, 0], scale)
                + F.cross_entropy(kernel_logits, kernel.long())
                + F.binary_cross_entropy_with_logits(jpeg_logits[:, 0], jpeg))
    jpeg_only = (F.smooth_l1_loss(quality_prediction[:, 0], quality, reduction="none")
                 + F.cross_entropy(sampling_logits, sampling.long(), reduction="none"))
    return (unmasked + (jpeg_only * jpeg).sum() / jpeg.sum().clamp_min(1)) / 5


def representation_loss(first, second, first_treatment, second_treatment):
    changed = (first_treatment != second_treatment).any(dim=1)
    distance = 1 - F.cosine_similarity(first.float(), second.float(), dim=1)
    return (distance * changed).sum() / changed.sum().clamp_min(1)


def joint_loss(model, first, second, labels):
    pixels, overlap, targets, active, anchors, treatments = (torch.cat((a, b)) for a, b in zip(first, second, strict=True))
    valid = overlap.diagonal(dim1=1, dim2=2) > 0
    patches = model.valid_embeddings(pixels, valid)
    logits, anchor_logits, forensic, representation, processing = model.training_outputs(patches, overlap, valid, anchors)
    supervised = grid_supervision(model, forensic, valid, targets, active, logits, labels, anchor_logits)
    stable = representation_loss(*representation.chunk(2), first[-1], second[-1])
    return supervised + TREATMENT_WEIGHT * processing_loss(processing, treatments) + REPRESENTATION_WEIGHT * stable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 100 * 60
    split, details = labeled_data()
    train, calibration = training_split(split["train"])
    tables, _ = tables_by_frequency(split["train"])
    seed_all(SEED, device)
    model = DegradationAnchorTexture().to(device, memory_format=torch.channels_last)
    report = {"variant": MODEL, "seed": SEED, "epochs": 24, "split_sha256": details["split_sha256"],
              "gpu": torch.cuda.get_device_name(device), "train_count": len(train),
              "calibration_count": len(calibration), "auxiliary_weight": AUXILIARY_WEIGHT,
              "treatment_weight": TREATMENT_WEIGHT, "representation_weight": REPRESENTATION_WEIGHT,
              "parameters": sum(p.numel() for p in model.parameters())}
    write_summary(args.output, report, "training")
    print(json.dumps(report), flush=True)
    views = training_views(train, tables[0], deadline)
    labels = torch.tensor([row.label for row in train])
    fit(model, views, labels, args.output, device, 24, deadline, variant=MODEL, get_batch=texture_batch,
        loss_fn=joint_loss)
    write_summary(args.output, report, "training_complete")


if __name__ == "__main__":
    main()

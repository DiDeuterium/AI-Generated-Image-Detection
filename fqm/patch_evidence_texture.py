"""Train a coverage-only control and a single detector that compares evidence across patches."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from math import log
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from balanced_texture_acquisition import BalancedAcquisition
from contrastive_texture_base import box_iou
from detection.data import jpeg_roundtrip
from detection.training import seed_all
from lqy.canonical_augmentation import CANONICAL
from lqy.codec_patch import encode, labeled_data, tables_by_frequency
from lqy.native import pad_to_crop
from lqy.night_train import CROP, check_time
from lqy.spectrum_patch import QUALITY
from lqy_resize_retrain import training_split
from native_texture_base import SEED, acquisition, consistency_loss, fit
from self_supervised_texture_base import (
    AUXILIARY_WEIGHT, TARGET_COUNT, SelfSupervisedTexture, residual_autocorrelation, texture_batch,
    training_assignments,
)
from two_stage_texture_base import write_summary


GRID_MODEL = "patch_grid_control"
EVIDENCE_MODEL = "patch_evidence_texture"
ANCHOR_CONTEXT_MODEL = "anchor_context_texture"
MULTIBAND_ANCHOR_MODEL = "multiband_anchor_context_texture"
ADAPTIVE_CONTEXT_MODEL = "adaptive_context_texture"
PATCH_COUNT = 9
MULTIBAND_CHANNELS = 8
ANCHOR_LOSS_WEIGHT = .25
MAX_CONTEXT_MARGIN = 2.0


class GridTexture(SelfSupervisedTexture):
    def valid_embeddings(self, pixels, valid):
        embeddings = self.patch_embeddings(pixels[valid].unsqueeze(1))
        padded = embeddings.new_zeros((*valid.shape, embeddings.shape[-1]))
        return padded.masked_scatter(valid[..., None], embeddings.flatten())

    def pool_valid(self, patches, valid):
        scores = self.attention(patches).float().squeeze(-1)
        weights = scores.masked_fill(~valid, -torch.inf).softmax(dim=1)
        mean = patches.masked_fill(~valid[..., None], 0).sum(dim=1) / valid.sum(dim=1, keepdim=True)
        focused = (patches * weights[..., None]).sum(dim=1)
        return .5 * (mean + focused)

    def classify_grid(self, patches, overlap, valid):
        return self.classifier(self.pool_valid(patches, valid))

    def forward(self, pixels, overlap):
        valid = overlap.diagonal(dim1=1, dim2=2) > 0
        return self.classify_grid(self.valid_embeddings(pixels, valid), overlap, valid)


class PatchEvidenceTexture(GridTexture):
    def __init__(self):
        super().__init__()
        patch_width = self.attention.in_features
        width = 64
        self.embedding = nn.Sequential(nn.LayerNorm(patch_width), nn.Linear(patch_width, width), nn.GELU())
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.context = nn.Linear(width, patch_width, bias=False)
        self.reliability = nn.Linear(2 * width + 1, 1)
        self.attention_scale = width ** -.5

    def classify_grid(self, patches, overlap, valid):
        tokens = self.embedding(patches)
        similarity = (self.query(tokens) @ self.key(tokens).transpose(1, 2)).float() * self.attention_scale
        relation = (similarity - overlap).masked_fill(~valid[:, None, :], -torch.inf).softmax(dim=-1)
        context = relation.to(tokens.dtype) @ tokens
        local_logits = self.classifier(patches)
        margin = local_logits[..., 1:] - local_logits[..., :1]
        reliability = self.attention(patches) + self.reliability(torch.cat((tokens, context, margin), dim=-1))
        reliability = reliability.float().squeeze(-1)
        # Dense clusters of overlapping patches contribute less total vote mass.
        redundancy = overlap.sum(dim=-1).clamp_min(1).log()
        weights = (reliability - redundancy).masked_fill(~valid, -torch.inf).softmax(dim=1)[..., None]
        local_vote = (weights * local_logits).sum(dim=1)
        pooled = (weights * (patches + self.context(context))).sum(dim=1)
        return .5 * (local_vote + self.classifier(pooled))


class AnchorContextTexture(GridTexture):
    def __init__(self):
        super().__init__()
        width = 64
        patch_width = self.attention.in_features
        self.context_embedding = nn.Sequential(nn.LayerNorm(patch_width), nn.Linear(patch_width, width), nn.GELU())
        self.context_query = nn.Linear(width, width, bias=False)
        self.context_correction = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.context_correction[-1].weight)
        nn.init.zeros_(self.context_correction[-1].bias)
        self.attention_scale = width ** -.5

    def context_strength(self, patches, anchor_logits, anchors, coverage, weights):
        return 1.0

    def classify_with_context(self, patches, overlap, valid, anchors):
        anchor_features = self.pool_valid(patches, anchors)
        anchor_logits = self.classifier(anchor_features)
        anchor = self.context_embedding(anchor_features)
        tokens = self.context_embedding(patches)
        extra = valid & ~anchors
        has_extra = extra.any(dim=1, keepdim=True)
        keys = extra | (~has_extra & anchors)
        query = self.context_query(anchor).unsqueeze(1)
        scores = (query @ tokens.transpose(1, 2)).squeeze(1).float() * self.attention_scale
        covered = overlap.masked_fill(~anchors[:, None, :], 0).amax(dim=-1)
        weights = (scores - covered).masked_fill(~keys, -torch.inf).softmax(dim=1)
        context = (tokens * weights[..., None]).sum(dim=1)
        inputs = torch.cat((anchor, context, context - anchor), dim=1)
        strength = self.context_strength(patches, anchor_logits, anchors, covered, weights)
        margin = MAX_CONTEXT_MARGIN * self.context_correction(inputs).float().tanh() * strength * has_extra
        return anchor_logits + .5 * torch.cat((-margin, margin), dim=1), anchor_logits

    def forward(self, pixels, overlap, anchors):
        valid = overlap.diagonal(dim1=1, dim2=2) > 0
        patches = self.valid_embeddings(pixels, valid)
        return self.classify_with_context(patches, overlap, valid, anchors)[0]


class AdaptiveContextTexture(AnchorContextTexture):
    def __init__(self):
        super().__init__()
        self.reliability_gate = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.reliability_gate[-1].weight)
        nn.init.constant_(self.reliability_gate[-1].bias, log(4))  # sigmoid = 0.8 at initialization

    def context_strength(self, patches, anchor_logits, anchors, coverage, weights):
        with torch.no_grad():
            # Use the same classification head without dropout for patch-level evidence.
            norm, hidden, activation, _, classifier = self.classifier
            logits = classifier(activation(hidden(norm(patches)))).float()
            evidence = (.5 * (logits[..., 1] - logits[..., 0])).tanh()
            anchor_weights = anchors.float() / anchors.sum(dim=1, keepdim=True)
            anchor_mean = (evidence * anchor_weights).sum(dim=1, keepdim=True)
            context_mean = (evidence * weights).sum(dim=1, keepdim=True)
            anchor_disagreement = ((evidence - anchor_mean).square() * anchor_weights).sum(dim=1, keepdim=True)
            context_disagreement = ((evidence - context_mean).square() * weights).sum(dim=1, keepdim=True)
            novelty = ((1 - coverage) * weights).sum(dim=1, keepdim=True)
            confidence = (anchor_logits[:, 1:] - anchor_logits[:, :1]).float().abs().tanh()
            stats = torch.cat((confidence, anchor_disagreement, context_disagreement, novelty,
                               (context_mean - anchor_mean).abs()), dim=1)
        return self.reliability_gate(stats).float().sigmoid()


def directional_band(image, step):
    """Haar-like directional responses, with a coefficient at every input pixel."""
    height, width = image.shape[-2:]
    padded = F.pad(image, (0, step, 0, step), mode="reflect")
    top_left = padded[..., :height, :width]
    top_right = padded[..., :height, step:]
    bottom_left = padded[..., step:, :width]
    bottom_right = padded[..., step:, step:]
    horizontal = .5 * (top_right - top_left + bottom_right - bottom_left)
    vertical = .5 * (bottom_left - top_left + bottom_right - top_right)
    diagonal = .5 * (top_left + bottom_right - top_right - bottom_left)
    return torch.cat((horizontal, vertical, diagonal), dim=1)


def multiband_residuals(rgb):
    luma = .299 * rgb[:, :1] + .587 * rgb[:, 1:2] + .114 * rgb[:, 2:3]
    smoothed = F.avg_pool2d(luma, 3, stride=1, padding=1, count_include_pad=False)

    def normalized(band):
        energy = F.avg_pool2d(band.square(), 5, stride=1, padding=2, count_include_pad=False)
        return (band * (energy + (4 / 255) ** 2).rsqrt()).clamp(-3, 3)

    fine = normalized(directional_band(luma, 1))
    mid = normalized(directional_band(smoothed, 2))
    coherence = (fine[:, :2] * mid[:, :2]).tanh()
    return torch.cat((fine, mid, coherence), dim=1)


class MultibandAnchorTexture(AnchorContextTexture):
    def __init__(self):
        super().__init__()
        original = self.features[0][0]
        expanded = nn.Conv2d(original.in_channels + MULTIBAND_CHANNELS, original.out_channels,
                             original.kernel_size, stride=original.stride, padding=original.padding, bias=False)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :original.in_channels].copy_(original.weight)
        self.features[0][0] = expanded

    def patch_features(self, views):
        features = super().patch_features(views)
        bands = multiband_residuals(features[:, :3].detach())
        return torch.cat((features, bands), dim=1)


def grid_layout(size):
    width, height = size
    right, bottom = max(0, width - CROP), max(0, height - CROP)
    # Dict order keeps the centre first and removes coordinates that collapse on small images.
    offsets = list(dict.fromkeys([(right // 2, bottom // 2)] + [
        (x, y) for y in (0, bottom // 2, bottom) for x in (0, right // 2, right)
    ]))
    pad_x, pad_y = max(0, CROP - width) // 2, max(0, CROP - height) // 2
    boxes = np.array([(max(0, x - pad_x), max(0, y - pad_y),
                       min(width, x + CROP - pad_x), min(height, y + CROP - pad_y)) for x, y in offsets])
    return offsets, boxes


def render_grid(image, table):
    offsets, boxes = grid_layout(image.size)
    canonical = pad_to_crop(jpeg_roundtrip(image, quality=CANONICAL, subsampling=2), CROP)
    pixels = np.zeros((PATCH_COUNT, 6, CROP, CROP), dtype=np.uint8)
    for index, (left, top) in enumerate(offsets):
        crop = canonical.crop((left, top, left + CROP, top + CROP))
        primary = jpeg_roundtrip(crop, quality=QUALITY, subsampling=2)
        pixels[index, :3] = np.asarray(primary).transpose(2, 0, 1)
        pixels[index, 3:] = np.asarray(encode(crop, table)).transpose(2, 0, 1)
    overlap = np.zeros((PATCH_COUNT, PATCH_COUNT), dtype=np.float32)
    overlap[:len(boxes), :len(boxes)] = box_iou(boxes[:, None], boxes[None, :])
    return pixels, overlap, boxes


def grid_input(row, table, processing):
    with Image.open(row["path"]) as source:
        image = source.convert("RGB")
    return render_grid(processing(image, row["image_id"]), table)[:2]


def anchor_roles(size):
    offsets, _ = grid_layout(size)
    right, bottom = max(0, size[0] - CROP), max(0, size[1] - CROP)
    anchors = {offsets[0], (0, 0), (right, 0), (0, bottom), (right, bottom)}
    mask = np.zeros(PATCH_COUNT, dtype=bool)
    mask[:len(offsets)] = [offset in anchors for offset in offsets]
    return mask


def anchor_grid_input(row, table, processing):
    with Image.open(row["path"]) as source:
        image = processing(source.convert("RGB"), row["image_id"])
    return (*render_grid(image, table)[:2], anchor_roles(image.size))


def grid_targets(image, rendered_size, boxes):
    width, height = rendered_size
    luma = np.asarray(image.convert("L"), dtype=np.float32)
    targets = np.zeros((PATCH_COUNT, TARGET_COUNT), dtype=np.float32)
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        left, top = x0 * image.width // width, y0 * image.height // height
        right = (x1 * image.width + width - 1) // width
        bottom = (y1 * image.height + height - 1) // height
        targets[index] = residual_autocorrelation(luma[top:bottom, left:right])
    return targets


def native_acquisitions(image, sample):
    for assignment in training_assignments(sample):
        yield acquisition(image, assignment), assignment[0] % 4 == 3


def training_pair(sample, table, include_anchors=False, acquisitions=native_acquisitions):
    with Image.open(sample.path) as source:
        image = source.convert("RGB")
    for processed, active in acquisitions(image, sample):
        pixels, overlap, boxes = render_grid(processed, table)
        targets = (grid_targets(image, processed.size, boxes) if active
                   else np.zeros((PATCH_COUNT, TARGET_COUNT), dtype=np.float32))
        view = (pixels, overlap, targets, active)
        yield (*view, anchor_roles(processed.size)) if include_anchors else view


def cache_training_views(samples, table, deadline, *, include_anchors=False, acquisitions=native_acquisitions):
    fields = (
        ((PATCH_COUNT, 6, CROP, CROP), np.uint8), ((PATCH_COUNT, PATCH_COUNT), np.float32),
        ((PATCH_COUNT, TARGET_COUNT), np.float32), ((), bool),
    )
    if include_anchors:
        fields += (((PATCH_COUNT,), bool),)
    storage = [[np.empty((len(samples), *shape), dtype=dtype) for shape, dtype in fields] for _ in range(2)]
    render = partial(training_pair, table=table, include_anchors=include_anchors, acquisitions=acquisitions)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(samples), 16):
            check_time(deadline)
            batch = workers.map(lambda sample: tuple(render(sample)), samples[start:start + 16])
            for row, pair in enumerate(batch, start=start):
                for arrays, view in zip(storage, pair, strict=True):
                    for array, value in zip(arrays, view, strict=True):
                        array[row] = value
            if (start + 16) % 640 == 0 or start + 16 >= len(samples):
                print(f"Cached patch grids for {min(start + 16, len(samples))}/{len(samples)} training images", flush=True)
    return tuple(tuple(map(torch.from_numpy, view)) for view in storage)


def grid_loss(model, first, second, labels):
    pixels, overlap, targets, active, *roles = (torch.cat((a, b)) for a, b in zip(first, second, strict=True))
    valid = overlap.diagonal(dim1=1, dim2=2) > 0
    patches = model.valid_embeddings(pixels, valid)
    if roles:
        logits, anchor_logits = model.classify_with_context(patches, overlap, valid, roles[0])
    else:
        logits = model.classify_grid(patches, overlap, valid)
        anchor_logits = None
    return grid_supervision(model, patches, valid, targets, active, logits, labels, anchor_logits)


def grid_supervision(model, patches, valid, targets, active, logits, labels, anchor_logits=None):
    a, b = logits.float().chunk(2)
    errors = F.smooth_l1_loss(model.texture_head(patches).float(), targets, reduction="none").mean(dim=2)
    per_image = (errors * valid).sum(dim=1) / valid.sum(dim=1)
    auxiliary = (per_image * active).sum() / active.sum().clamp_min(1)
    supervised = consistency_loss(a, b, labels)
    if anchor_logits is not None:
        anchor_loss = consistency_loss(*anchor_logits.float().chunk(2), labels)
        supervised = (1 - ANCHOR_LOSS_WEIGHT) * supervised + ANCHOR_LOSS_WEIGHT * anchor_loss
    return supervised + AUXILIARY_WEIGHT * auxiliary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-context-only", action="store_true")
    parser.add_argument("--balanced-acquisition", action="store_true")
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument("--multiband", action="store_true")
    variants.add_argument("--adaptive-context", action="store_true")
    args = parser.parse_args()
    if args.multiband and not args.anchor_context_only:
        parser.error("--multiband requires --anchor-context-only")
    if args.adaptive_context and not args.anchor_context_only:
        parser.error("--adaptive-context requires --anchor-context-only")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 135 * 60
    split, details = labeled_data()
    train, calibration = training_split(split["train"])
    tables, _ = tables_by_frequency(split["train"])
    acquisitions = BalancedAcquisition(train) if args.balanced_acquisition else native_acquisitions
    if args.balanced_acquisition:
        print("Matched training treatments per class:", dict(acquisitions.counts), flush=True)
    views = cache_training_views(train, tables[0], deadline, include_anchors=args.anchor_context_only,
                                 acquisitions=acquisitions)
    labels = torch.tensor([row.label for row in train])
    arms = (("grid_control", GRID_MODEL, GridTexture), ("patch_evidence", EVIDENCE_MODEL, PatchEvidenceTexture))
    if args.anchor_context_only:
        arms = (("anchor_context", ANCHOR_CONTEXT_MODEL, AnchorContextTexture),)
    if args.multiband:
        arms = (("anchor_context", MULTIBAND_ANCHOR_MODEL, MultibandAnchorTexture),)
    if args.adaptive_context:
        arms = (("anchor_context", ADAPTIVE_CONTEXT_MODEL, AdaptiveContextTexture),)
    for arm, variant, model_class in arms:
        seed_all(SEED, device)
        model = model_class().to(device, memory_format=torch.channels_last)
        output = args.output / arm
        output.mkdir()
        report = {"variant": variant, "seed": SEED, "epochs": 24, "split_sha256": details["split_sha256"],
                  "gpu": torch.cuda.get_device_name(device), "train_count": len(train),
                  "calibration_count": len(calibration), "max_patches": PATCH_COUNT,
                  "auxiliary_weight": AUXILIARY_WEIGHT, "parameters": sum(p.numel() for p in model.parameters())}
        if args.anchor_context_only:
            report.update(anchor_loss_weight=ANCHOR_LOSS_WEIGHT, max_context_margin=MAX_CONTEXT_MARGIN)
        if args.multiband:
            report["multiband_channels"] = MULTIBAND_CHANNELS
        if args.adaptive_context:
            report["context_gate"] = "patch_disagreement_and_coverage"
        if args.balanced_acquisition:
            report["treatment_counts_per_class"] = dict(acquisitions.counts)
        write_summary(output, report, "training")
        print(json.dumps(report), flush=True)
        fit(model, views, labels, output, device, 24, deadline, variant=variant, get_batch=texture_batch,
            loss_fn=grid_loss)
        write_summary(output, report, "training_complete")


if __name__ == "__main__":
    main()

"""Jointly learn classification, residual autocorrelation, and cross-degradation patch matching."""

import argparse
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detection.training import seed_all
from lqy.canonical_augmentation import TRAIN_CROPS
from lqy.codec_patch import labeled_data, tables_by_frequency
from lqy_resize_retrain import training_split
from native_texture_base import SEED, consistency_loss, fit
from self_supervised_texture_base import (
    AUXILIARY_WEIGHT, SelfSupervisedTexture, patch_boxes, residual_autocorrelation,
    texture_batch, texture_loss, training_assignments, training_views,
)


MODEL = "contrastive_texture_base"
CLASS_CONTRASTIVE_MODEL = "real_anchored_contrastive_texture"
CROSS_CLASS_CONTRASTIVE_MODEL = "cross_class_contrastive_texture"
MOMENTUM_CONTRASTIVE_MODEL = "momentum_contrastive_texture"
CONTRASTIVE_WEIGHT = .05
TEMPERATURE = .20
MIN_OVERLAP = .50
MOMENTUM = .99
QUEUE_SIZE = 1024


class ContrastiveTexture(SelfSupervisedTexture):
    def __init__(self):
        super().__init__()
        width = self.attention.in_features
        self.projector = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 64))

    def training_outputs(self, pixels):
        patches = self.texture_embeddings(pixels)
        logits = self.classifier(self.pool_embeddings(patches))
        return logits, self.texture_head(patches), self.projector(patches)


def box_iou(first, second):
    shared = np.maximum(0, np.minimum(first[..., 2:], second[..., 2:])
                        - np.maximum(first[..., :2], second[..., :2])).prod(axis=-1)
    first_area = (first[..., 2:] - first[..., :2]).prod(axis=-1)
    second_area = (second[..., 2:] - second[..., :2]).prod(axis=-1)
    return shared / (first_area + second_area - shared)


def contrastive_views(samples, table, deadline):
    positions = {}
    conditions = np.empty((2, len(samples)), dtype=np.uint8)
    boxes = np.full((2, len(samples), len(TRAIN_CROPS), 4), np.nan, dtype=np.float32)
    for index, row in enumerate(samples):
        for view, assignment in enumerate(training_assignments(row)):
            positions[assignment] = view, index
            conditions[view, index] = assignment[0] % 4
    if len(positions) != 2 * len(samples):
        raise ValueError("Expected a unique assignment for each training view")

    def record_view(size, assignment):
        width, height = size
        # Resizing preserves normalized coordinates. The boxes exclude reflected padding.
        boxes[positions[assignment]] = np.asarray(list(patch_boxes(size))) / (width, height, width, height)

    views = training_views(samples, table, deadline, target_fn=residual_autocorrelation, record_view=record_view)
    if not np.isfinite(boxes).all():
        raise ValueError("Missing geometry for one or more training views")
    first, second = conditions
    cross_degradation = ((first == 3) & (second <= 1)) | ((second == 3) & (first <= 1))
    positive_pairs = torch.from_numpy(cross_degradation[:, None] & (box_iou(*boxes) >= MIN_OVERLAP))
    return tuple((*view, positive_pairs) for view in views), positive_pairs


def similarity_logits(embeddings):
    normalized = F.normalize(embeddings.float(), dim=-1).transpose(0, 1)
    return (normalized @ normalized.transpose(1, 2)).float() / TEMPERATURE


def patch_contrastive_loss(embeddings, matched, labels=None):
    # Only the same crop position is compared across images. Self-similarities are excluded.
    batch = len(matched)
    logits = similarity_logits(embeddings)
    anchors = torch.arange(2 * batch, device=embeddings.device)
    positive = (anchors + batch) % (2 * batch)
    if labels is None:
        candidates = anchors[:, None] != anchors[None, :]
    else:
        classes = labels.repeat(2)
        candidates = (classes[:, None] != classes[None, :]) | (positive[:, None] == anchors[None, :])
    return masked_patch_loss(logits, candidates, positive, matched)


def masked_patch_loss(logits, candidates, positive, matched):
    logits = logits.masked_fill(~candidates, -torch.inf)
    losses = F.cross_entropy(logits.flatten(0, 1), positive.repeat(matched.shape[1]), reduction="none")
    active = matched.repeat(2, 1).T.reshape(-1)
    return (losses * active).sum() / active.sum().clamp_min(1)


def real_anchored_loss(embeddings, labels, conditions):
    real = labels.repeat(2) == 0
    mild, strong = conditions <= 1, conditions == 3
    image_ids = torch.arange(len(labels), device=labels.device).repeat(2)
    cross_degradation = (mild[:, None] & strong[None, :]) | (strong[:, None] & mild[None, :])
    candidates = cross_degradation & (image_ids[:, None] != image_ids[None, :])
    positives = candidates & real[None, :]
    negatives = candidates & ~real[None, :]
    valid = real & positives.any(dim=1) & negatives.any(dim=1)

    logits = similarity_logits(embeddings)
    # Give inactive anchors a finite denominator, then omit their losses.
    denominator_mask = candidates | ~valid[:, None]
    log_denominator = logits.masked_fill(~denominator_mask, -torch.inf).logsumexp(dim=-1)
    positive_mean = logits.masked_fill(~positives, 0).sum(dim=-1) / positives.sum(dim=-1).clamp_min(1)
    losses = (log_denominator - positive_mean).masked_fill(~valid, 0)
    return losses.mean(dim=0).sum() / valid.sum().clamp_min(1)


def class_contrastive_views(samples, table, deadline):
    conditions = torch.tensor([[assignment[0] % 4 for assignment in training_assignments(row)]
                               for row in samples], dtype=torch.uint8)
    views = training_views(samples, table, deadline, target_fn=residual_autocorrelation)
    return tuple((*view, condition) for view, condition in zip(views, conditions.unbind(dim=1), strict=True))


def paired_tensors(first, second):
    return tuple(torch.cat((a, b)) for a, b in zip(first[:3], second[:3], strict=True))


def classification_and_texture(model, pixels, targets, active, labels):
    logits, reconstruction, embeddings = model.training_outputs(pixels)
    a, b = logits.float().chunk(2)
    supervised = consistency_loss(a, b, labels)
    autocorrelation = texture_loss(reconstruction, targets, active)
    return supervised + AUXILIARY_WEIGHT * autocorrelation, embeddings


def joint_contrastive_loss(model, first, second, labels, *, pairing="instance"):
    main_loss, embeddings = classification_and_texture(model, *paired_tensors(first, second), labels)
    if pairing == "real-anchored":
        contrastive = real_anchored_loss(embeddings, labels, torch.cat((first[3], second[3])))
    else:
        contrastive = patch_contrastive_loss(embeddings, first[3], labels if pairing == "cross-class" else None)
    return main_loss + CONTRASTIVE_WEIGHT * contrastive


def momentum_batch(view, ids, device):
    return (*texture_batch(view, ids, device), ids.to(device))


class MomentumContrast:
    """Compute contrast against momentum keys from the batch and a label-free FIFO queue."""

    def __init__(self, model):
        self.encoder = deepcopy(model).requires_grad_(False).eval()
        self.queued_keys = None
        self.queued_ids = None

    @torch.no_grad()
    def project_keys(self, model, pixels):
        for current, momentum in zip(model.parameters(), self.encoder.parameters(), strict=True):
            momentum.lerp_(current, 1 - MOMENTUM)
        for current, momentum in zip(model.buffers(), self.encoder.buffers(), strict=True):
            momentum.copy_(current)
        embeddings = self.encoder.projector(self.encoder.texture_embeddings(pixels))
        return F.normalize(embeddings.float(), dim=-1).transpose(0, 1)

    def contrast(self, embeddings, keys, ids, matched):
        queries = F.normalize(embeddings.float(), dim=-1).transpose(0, 1)
        if self.queued_keys is None:
            candidates, candidate_ids = keys, ids
        else:
            candidates = torch.cat((keys, self.queued_keys), dim=1)
            candidate_ids = torch.cat((ids, self.queued_ids))
        logits = (queries @ candidates.transpose(1, 2)).float() / TEMPERATURE
        positive = (torch.arange(len(ids), device=ids.device) + len(matched)) % len(ids)
        eligible = (ids[:, None] != candidate_ids[None, :]) | (
            positive[:, None] == torch.arange(len(candidate_ids), device=ids.device)[None, :])
        return masked_patch_loss(logits, eligible, positive, matched)

    def enqueue(self, keys, ids):
        if self.queued_keys is not None:
            keys = torch.cat((self.queued_keys, keys), dim=1)
            ids = torch.cat((self.queued_ids, ids))
        self.queued_keys = keys[:, -QUEUE_SIZE:].detach()
        self.queued_ids = ids[-QUEUE_SIZE:].detach()

    def __call__(self, model, first, second, labels):
        pixels, targets, active = paired_tensors(first, second)
        keys = self.project_keys(model, pixels)
        main_loss, embeddings = classification_and_texture(model, pixels, targets, active, labels)
        ids = torch.cat((first[4], second[4]))
        contrastive = self.contrast(embeddings, keys, ids, first[3])
        self.enqueue(keys, ids)
        return main_loss + CONTRASTIVE_WEIGHT * contrastive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairing", choices=("instance", "real-anchored", "cross-class", "momentum"), default="instance")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 75 * 60
    split, details = labeled_data()
    train, calibration = training_split(split["train"])
    tables, _ = tables_by_frequency(split["train"])
    seed_all(SEED, device)
    model = ContrastiveTexture().to(device, memory_format=torch.channels_last)
    class_pairs = args.pairing == "real-anchored"
    variant = {"instance": MODEL, "real-anchored": CLASS_CONTRASTIVE_MODEL,
               "cross-class": CROSS_CLASS_CONTRASTIVE_MODEL, "momentum": MOMENTUM_CONTRASTIVE_MODEL}[args.pairing]
    report = {"variant": variant, "seed": SEED, "epochs": 24, "split_sha256": details["split_sha256"],
              "gpu": torch.cuda.get_device_name(device), "train_count": len(train),
              "calibration_count": len(calibration), "auxiliary_weight": AUXILIARY_WEIGHT,
              "contrastive_weight": CONTRASTIVE_WEIGHT, "temperature": TEMPERATURE,
              "contrastive_pairing": args.pairing, "stage": "training"}
    if not class_pairs:
        report["min_pair_overlap_iou"] = MIN_OVERLAP
    if args.pairing == "momentum":
        report.update(momentum=MOMENTUM, queue_size=QUEUE_SIZE)
    summary = args.output / "summary.json"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if class_pairs:
        views = class_contrastive_views(train, tables[0], deadline)
    else:
        views, pairs = contrastive_views(train, tables[0], deadline)
        report["contrastive_image_pairs"] = int(pairs.any(dim=1).sum())
        report["contrastive_patch_pairs"] = int(pairs.sum())
        print(f"Matched {report['contrastive_patch_pairs']} patch pairs across {report['contrastive_image_pairs']} images",
              flush=True)
    batch_fn, loss_fn = texture_batch, partial(joint_contrastive_loss, pairing=args.pairing)
    if args.pairing == "momentum":
        batch_fn, loss_fn = momentum_batch, MomentumContrast(model)
    fit(model, views, torch.tensor([row.label for row in train]), args.output, device, 24, deadline,
        variant=variant, get_batch=batch_fn, loss_fn=loss_fn)
    report["stage"] = "training_complete"
    summary.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

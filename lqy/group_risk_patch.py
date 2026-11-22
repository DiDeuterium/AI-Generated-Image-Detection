"""Train a native-patch CNN against hard compression and image-size groups."""

import argparse
from collections import Counter
from functools import partial
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from detection.training import write_json

from .cnn import device_for, run_lock
from .group_patch import ARCHITECTURE
from .night_train import CROP, SEEDS, SCREEN_EPOCHS, fit
from .phase_patch import run
from .robust import labeled_splits


RULES = {"group_risk": (ARCHITECTURE, 1.0), "group_risk_reference_mean": (ARCHITECTURE, 0.5)}


def real_compression_groups():
    args = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    split, _ = labeled_splits(args, {"train_per_class": 3200, "val_per_class": 800,
                                     "short_side": CROP, "crop_size": CROP})
    tables = {}
    for row, sample in enumerate(split["train"]):
        if sample.label == 0:
            with Image.open(sample.path) as image:
                tables[row] = tuple((i, tuple(values)) for i, values in sorted(image.quantization.items()))
    dominant, _ = Counter(tables.values()).most_common(1)[0]
    return {row: int(key != dominant) for row, key in tables.items()}


class GroupRisk:
    def __init__(self, groups, device):
        self.groups = groups
        self.loss = torch.zeros(6, device=device)
        self.class_blocks = (slice(2), slice(2, 6))

    def __call__(self, logits, targets, rows):
        values = F.cross_entropy(logits.float(), targets, reduction="none")
        groups = self.groups[rows].to(logits.device)
        membership = F.one_hot(groups, num_classes=6).float()
        counts = membership.sum(dim=0)
        active = counts > 0
        group_mean = (membership * values[:, None]).sum(dim=0) / counts.clamp(min=1)
        with torch.no_grad():
            self.loss.lerp_(torch.where(active, group_mean.detach(), self.loss), 0.03)
        # Keep separate real and AI contributions. The ordinary loss remains
        # half of the objective so a transient high-loss group cannot dominate.
        robust = []
        for block in self.class_blocks:
            weight = (2 * self.loss[block].clamp(max=2)).exp() * active[block]
            robust.append((weight * group_mean[block]).sum() / weight.sum().clamp(min=1))
        present_classes = active[:2].any().float() + active[2:].any().float()
        return 0.5 * (values.mean() + sum(robust) / present_classes.clamp(min=1))


def train_variant(folder, variant, views, labels, sizes, rows, device, deadline, *, real_groups):
    groups = torch.zeros_like(labels)
    real = [i for i in rows.tolist() if labels[i] == 0]
    fake = [i for i in rows.tolist() if labels[i] == 1]
    groups[real] = torch.tensor([real_groups[i] for i in real])
    sides = [min(sizes[i]) for i in fake]
    groups[fake] = torch.from_numpy(2 + np.searchsorted((192, 384, 768), sides, side="right"))
    write_json(folder / "training_groups.json", {"real_quantization_and_ai_size_counts":
                                                  torch.bincount(groups[rows], minlength=6).tolist()})
    models = []
    for seed in SEEDS:
        objective = GroupRisk(groups, device)
        models.append(fit(folder / f"s{seed}", variant, seed, SCREEN_EPOCHS, views, labels, rows, device, deadline,
                          objective=objective))
        write_json(folder / f"s{seed}" / "group_losses.json", {"final_exponential_mean": objective.loss.cpu().tolist()})
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        trainer = partial(train_variant, real_groups=real_compression_groups())
        run(args, device_for(args.device, 6), time.monotonic() + 3600, rules=RULES, train_model=trainer)


if __name__ == "__main__":
    main()

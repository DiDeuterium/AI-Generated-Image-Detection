"""Train a patch CNN that pools evidence from several subtypes per class."""

import argparse
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detection.training import write_json

from .cnn import device_for, run_lock
from .group_patch import ARCHITECTURE
from .night_train import SEEDS, SCREEN_EPOCHS, PatchCNN, fit
from .phase_patch import run


RULES = {"subtypes": ("size_subtypes", 1.0), "subtypes_reference_mean": ("size_subtypes", 0.5)}


def training_targets(sizes, labels, rows):
    sides = np.array([min(sizes[index]) for index in rows.tolist()])
    fake = labels[rows].numpy() == 1
    cuts = np.unique(np.quantile(sides[~fake], (0.25, 0.5, 0.75)))
    subtypes = np.searchsorted(cuts, sides, side="right")
    subtypes[fake] = 4 + np.searchsorted((192, 384, 768), sides[fake], side="right")
    counts = np.bincount(subtypes, minlength=8)
    targets = torch.zeros_like(labels)
    targets[rows] = torch.from_numpy(subtypes)
    return targets, counts, cuts


class SubtypePatchCNN(PatchCNN):
    def __init__(self, _variant, active):
        super().__init__(ARCHITECTURE)
        self.classifier = nn.Linear(128, 8)
        self.register_buffer("active", active.reshape(2, 4))

    def subtype_scores(self, views):
        features = self.patch_embeddings(views)
        return self.classifier(features).reshape(len(views), views.shape[1], 2, 4).mean(dim=1).float()

    def binary_logits(self, subtype_scores):
        masked = subtype_scores.masked_fill(~self.active, -torch.inf)
        return masked.logsumexp(dim=-1) - self.active.sum(dim=-1).float().log()

    def subtype_loss(self, subtype_scores, labels, targets):
        # The auxiliary term distinguishes subtypes only inside the true class.
        own_class = subtype_scores[torch.arange(len(labels), device=labels.device), labels]
        return F.cross_entropy(own_class.masked_fill(~self.active[labels], -torch.inf), targets.remainder(4))

    def forward(self, views):
        return self.binary_logits(self.subtype_scores(views))


def train_variant(folder, variant, views, labels, sizes, rows, device, deadline):
    targets, counts, cuts = training_targets(sizes, labels, rows)
    write_json(folder / "subtype_groups.json", {"real_short_edge_thresholds": cuts.tolist(),
                                                "training_counts": counts.tolist()})
    factory = partial(SubtypePatchCNN, active=torch.as_tensor(counts > 0))
    return [fit(folder / f"s{seed}", variant, seed, SCREEN_EPOCHS, views, labels, rows, device, deadline,
                model_factory=factory, subtype_targets=targets) for seed in SEEDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600, rules=RULES, train_model=train_variant)


if __name__ == "__main__":
    main()

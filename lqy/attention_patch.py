"""Test learned weighting of the five native image patches."""

import argparse
import time
from pathlib import Path

from torch import nn

from .cnn import device_for, run_lock
from .group_patch import ARCHITECTURE
from .night_train import SEEDS, SCREEN_EPOCHS, PatchCNN, fit
from .phase_patch import run


RULES = {"attention": ("attention", 1.0), "attention_reference_mean": ("attention", 0.5)}


class AttentionPatchCNN(PatchCNN):
    def __init__(self, _variant):
        super().__init__(ARCHITECTURE)
        self.attention = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 24), nn.Tanh(), nn.Linear(24, 1, bias=False))
        nn.init.zeros_(self.attention[-1].weight)

    def forward(self, views):
        features = self.patch_embeddings(views).reshape(len(views), views.shape[1], -1)
        logits = self.classifier(features)
        weights = self.attention(features).float().softmax(dim=1)
        # A small equal-weight contribution keeps every patch represented.
        weights = 0.85 * weights + 0.15 / features.shape[1]
        return (logits * weights).sum(dim=1)


def train_variant(folder, variant, views, labels, _sizes, rows, device, deadline):
    return [fit(folder / f"s{seed}", variant, seed, SCREEN_EPOCHS, views, labels, rows, device, deadline,
                model_factory=AttentionPatchCNN) for seed in SEEDS]


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

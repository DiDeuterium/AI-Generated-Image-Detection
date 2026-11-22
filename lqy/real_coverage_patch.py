"""Keep all training photographs and give underrepresented real-image sizes mild weight."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from detection.training import write_json

from .cnn import device_for, run_lock
from .group_patch import ARCHITECTURE
from .night_train import SEEDS, SCREEN_EPOCHS, fit
from .phase_patch import run


RULES = {"real_coverage": (ARCHITECTURE, 1.0), "real_coverage_reference_mean": (ARCHITECTURE, 0.5)}
SIDES = (384, 513)


def loss_weights(rows, labels, sizes):
    labels_train = labels[rows].numpy()
    real = labels_train == 0
    real_sizes = np.asarray([min(sizes[i]) for i in rows[torch.as_tensor(real)].tolist()])
    bins = np.searchsorted(SIDES, real_sizes, side="right")
    counts = np.bincount(bins, minlength=len(SIDES) + 1)
    real_weights = 1 / np.sqrt(counts[bins])
    # Equal total loss weight for the two classes, with no within-AI weighting.
    target = len(rows) * 0.5
    real_weights *= target / real_weights.sum()
    cohort_weights = np.empty(len(rows), dtype=np.float32)
    cohort_weights[real] = real_weights
    cohort_weights[~real] = target / (~real).sum()
    weights = torch.zeros_like(labels, dtype=torch.float32)
    weights[rows] = torch.from_numpy(cohort_weights)
    return weights, {"real_counts_by_short_side": counts.tolist(), "ai_count": int((~real).sum()),
                     "short_side_thresholds": SIDES}


def train_variant(folder, variant, views, labels, sizes, rows, device, deadline):
    weights, metadata = loss_weights(rows, labels, sizes)
    write_json(folder / "training_counts.json", metadata)
    print(f"{folder.parent.name}: training counts {metadata}", flush=True)
    return [fit(folder / f"s{seed}", variant, seed, SCREEN_EPOCHS, views, labels, rows, device, deadline,
                example_weights=weights) for seed in SEEDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600, rules=RULES, train_model=train_variant,
            include_all_real=True)


if __name__ == "__main__":
    main()

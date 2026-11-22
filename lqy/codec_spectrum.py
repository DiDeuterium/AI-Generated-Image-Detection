"""Train the frequency branch with matched alternate JPEG encodes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import time

import numpy as np
import torch

from detection.training import write_json

from .cnn import device_for, run_lock
from .codec_patch import labeled_data, prepare, tables_by_frequency
from .night_train import CROP, SEEDS, SCREEN_EPOCHS, PatchCNN, check_time, fit
from .phase_patch import run
from .spectrum_patch import spectrum_channels


VARIANT = "codec_spectrum"
RULES = {VARIANT: (VARIANT, 1.0), "replace_frequency": (VARIANT, 0.5)}


class FrequencyCNN(PatchCNN):
    def __init__(self, variant):
        if variant != VARIANT:
            raise ValueError(f"Unknown frequency variant: {variant}")
        super().__init__("residual")

    def patch_features(self, views):
        return views.flatten(0, 1).float() / 255 - 0.5


def cache_features(items, transform, deadline):
    result = np.empty((len(items), 5, 3, CROP, CROP), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(items), 32):
            check_time(deadline)
            for i, view in enumerate(workers.map(transform, items[start:start + 32]), start=start):
                result[i] = view
    return torch.from_numpy(result)


def alternate_spectrum(job):
    return spectrum_channels(prepare(job))


def standard_features(_samples, views, deadline):
    print(f"Computing Q75 spectra for {len(views)} images", flush=True)
    return cache_features(views.numpy(), spectrum_channels, deadline)


def train_variant(folder, variant, views, labels, _sizes, rows, device, deadline, *, paired):
    return [fit(folder / f"s{seed}", variant, seed, SCREEN_EPOCHS, views, labels, rows,
                device, deadline, model_factory=FrequencyCNN, paired_data=paired) for seed in SEEDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        deadline = time.monotonic() + 3600
        split, _ = labeled_data()
        profiles, description = tables_by_frequency(split["train"])
        write_json(args.output / "profiles.json", description[0])
        jobs = [(sample, profiles[0]) for sample in split["train"]]
        print(f"Computing matched alternate JPEG spectra for {len(jobs)} training images", flush=True)
        paired = cache_features(jobs, alternate_spectrum, deadline)
        run(args, device_for(args.device, 6), deadline, rules=RULES, train_model=partial(train_variant, paired=paired),
            candidate_data=standard_features, blend_with_spatial=True)


if __name__ == "__main__":
    main()

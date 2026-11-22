"""Compare a scratch CNN that analyzes the pixel response to two JPEG encodes."""

import argparse
from functools import partial
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional as F

from detection.training import write_json

from .cnn import device_for, run_lock
from .codec_patch import cache_encodes, labeled_data, tables_by_frequency
from .group_patch import ARCHITECTURE
from .night_train import PatchCNN, SEEDS, SCREEN_EPOCHS, fit
from .phase_patch import run


VARIANT = "codec_response"
RULES = {VARIANT: (VARIANT, 1.0), "codec_response_reference_mean": (VARIANT, 0.5)}


class PairedViews:
    def __init__(self, reference, alternate):
        self.reference = reference
        self.alternate = alternate

    def __getitem__(self, rows):
        return torch.cat((self.reference[rows], self.alternate[rows]), dim=-3)


def candidate_data(samples, views, deadline, *, table):
    print(f"Encoding a second fixed JPEG view for {len(samples)} images", flush=True)
    return PairedViews(views, cache_encodes(samples, [table] * len(samples), deadline))


class CodecResponseCNN(PatchCNN):
    def __init__(self, variant):
        if variant != VARIANT:
            raise ValueError(f"Unknown JPEG response variant: {variant}")
        super().__init__(ARCHITECTURE)
        self.features[0] = nn.Conv2d(9, 16, 5, stride=4, padding=2, bias=False)

    def patch_features(self, views):
        q75, alternate = (views.flatten(0, 1).float() / 255).split(3, dim=1)
        common = (q75 + alternate) * 0.5
        blur = F.avg_pool2d(common, 3, stride=1, padding=1, count_include_pad=False)
        return torch.cat((common - 0.5, 6 * (common - blur), 6 * (alternate - q75)), dim=1)


def train_variant(folder, variant, views, labels, _sizes, rows, device, deadline, *, model_factory=CodecResponseCNN,
                  paired_data=None):
    return [fit(folder / f"s{seed}", variant, seed, SCREEN_EPOCHS, views, labels, rows,
                device, deadline, model_factory=model_factory, paired_data=paired_data) for seed in SEEDS]


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
        run(args, device_for(args.device, 6), deadline, rules=RULES, train_model=train_variant,
            candidate_data=partial(candidate_data, table=profiles[0]))


if __name__ == "__main__":
    main()

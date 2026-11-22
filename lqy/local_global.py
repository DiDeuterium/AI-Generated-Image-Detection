"""Train a patch CNN with an additional low-resolution whole-image branch."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn

from detection.data import read_test
from detection.metrics import binary_metrics
from detection.training import write_csv, write_json

from .cnn import device_for, run_lock
from .group_patch import ARCHITECTURE, restore_ensemble
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, PatchCNN, check_time, fit, image_patches, sample_rows, scores
from .robust import labeled_splits
from .spectrum_patch import CONFIRM, QUALITY, SCREEN, accuracy_interval


VARIANT = "local_global"
GLOBAL_SIZE = CROP // 2


class LocalGlobalCNN(PatchCNN):
    def __init__(self, variant):
        if variant != VARIANT:
            raise ValueError(f"Unknown local-global variant {variant}")
        super().__init__(ARCHITECTURE)
        self.scene = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 40, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(40), nn.ReLU(),
            nn.Conv2d(40, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(64 * 16, 64), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(64, 2),
        )

    def forward(self, views):
        patches = super().forward(views[:, :5])
        entire_image = views[:, 5, :, ::2, ::2].float() / 255 - 0.5
        composition = self.scene(entire_image.contiguous(memory_format=torch.channels_last))
        return patches + 0.5 * composition


def prepare(sample):
    with Image.open(sample.path) as source:
        size = source.size
        rgb = source.convert("RGB")
    local = image_patches(rgb, QUALITY)
    entire = rgb.resize((GLOBAL_SIZE, GLOBAL_SIZE), Image.Resampling.LANCZOS)
    # Pack all views into one tensor. The global branch reads the original 64x64 samples.
    packed = np.asarray(entire.resize((CROP, CROP), Image.Resampling.NEAREST)).transpose(2, 0, 1)
    return size, np.concatenate((local, packed[None]), axis=0)


def cache(samples, deadline):
    print(f"Preparing native crops and whole-image views for {len(samples)} images", flush=True)
    values = np.empty((len(samples), 6, 3, CROP, CROP), dtype=np.uint8)
    sizes = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(samples), 32):
            check_time(deadline)
            for i, (size, tensor) in enumerate(executor.map(prepare, samples[start:start + 32]), start=start):
                values[i] = tensor
                sizes.append(size)
    return torch.from_numpy(values), sizes


def run(args, device, deadline):
    data = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    split, identity = labeled_splits(data, {"train_per_class": 3200, "val_per_class": 800,
                                           "short_side": CROP, "crop_size": CROP})
    samples = split["train"] + split["val"]
    labels = torch.tensor([sample.label for sample in samples])
    train_end = len(split["train"])
    report = {"epochs": SCREEN_EPOCHS, "seeds": SEEDS, "local_jpeg_quality": QUALITY,
              "whole_image_side": GLOBAL_SIZE, "split_sha256": identity["split_sha256"], "results": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("prepare_labeled")
    packed, sizes = cache(samples, deadline)
    local = packed[:, :5]
    real = sorted((i for i in range(train_end, len(samples)) if samples[i].label == 0),
                  key=lambda i: sha256(f"threshold-audit:{samples[i].image_id}".encode()).digest())
    offset = 0

    def train(folder, rows, with_global=True):
        constructor = LocalGlobalCNN if with_global else PatchCNN
        variant = VARIANT if with_global else ARCHITECTURE
        source = packed if with_global else local
        return [fit(folder / f"s{seed}", variant, seed, SCREEN_EPOCHS, source, labels, rows, device,
                    deadline, model_factory=constructor) for seed in SEEDS]

    def fold(cohort):
        nonlocal offset
        train_rows, group = sample_rows(samples, sizes, train_end, cohort)
        fake = group[len(group) // 2:].tolist()
        rows = torch.tensor(real[offset:offset + len(fake)] + fake)
        if len(rows) != 2 * len(fake):
            raise ValueError("Insufficient disjoint real validation samples")
        offset += len(fake)
        trial = train(args.output / f"local_global_{cohort}", train_rows)
        if cohort == CONFIRM:
            reference = train(args.output / "reference_128", train_rows, with_global=False)
        else:
            folder = (args.group_reference / f"confirm_{cohort}_reference" if cohort == 256
                      else args.night_reference / f"screen_{cohort}_{ARCHITECTURE}")
            reference = restore_ensemble([folder / f"s{seed}" / "last.pt" for seed in SEEDS], device)
        truth = labels[rows].numpy()
        first = scores(reference, local, rows, device, deadline)
        second = scores(trial, packed, rows, device, deadline)
        a = binary_metrics(truth, first > 0.5, first)
        b = binary_metrics(truth, second > 0.5, second)
        print(f"{cohort}: accuracy {a['accuracy']:.3%} -> {b['accuracy']:.3%}; "
              f"AUROC {a['auroc']:.4f} -> {b['auroc']:.4f}", flush=True)
        return {"reference": a, "local_global": b,
                "delta": {key: b[key] - a[key] for key in ("accuracy", "auroc")},
                "paired_accuracy": accuracy_interval(truth, first, second)}

    for cohort in SCREEN:
        record(f"screen_{cohort}")
        report["results"][str(cohort)] = fold(cohort)
    wins = sum(all(change > 0 for change in item["delta"].values()) for item in report["results"].values())
    report["groups_improved_on_both_metrics"] = wins
    if wins >= 2:
        record("confirm_128")
        report["confirmation"] = fold(CONFIRM)

    record("full_training")
    train_rows, val_rows = sample_rows(samples, sizes, train_end)
    models = train(args.output / "full", train_rows)
    probability = scores(models, packed, val_rows, device, deadline)
    report["full_validation"] = binary_metrics(labels[val_rows].tolist(), probability > 0.5, probability)
    report["train_count"] = len(train_rows)
    del local, packed

    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"),
                     Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    views, _ = cache(test, deadline)
    prediction = scores(models, views, torch.arange(len(test)), device, deadline) > 0.5
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(label)) for sample, label in zip(test, prediction, strict=True)))
    record("complete")
    print(f"Wrote {len(test)} predictions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--group-reference", type=Path, default=Path("lqy/runs/group-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 1800)


if __name__ == "__main__":
    main()

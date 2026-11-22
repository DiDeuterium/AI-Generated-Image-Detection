"""Classify joint distributions of local pixel residuals on native crops."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from detection.data import read_test
from detection.metrics import binary_metrics
from detection.training import write_csv, write_json

from .cnn import device_for, run_lock
from .group_patch import ARCHITECTURE, restore_ensemble
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, cache as rgb_cache, check_time, fit, prepare_image, sample_rows, scores
from .robust import labeled_splits
from .spectrum_patch import QUALITY, SCREEN, accuracy_interval


VARIANT = "residual_cooccurrence"
BINS = 7
CHANNELS = 3 * 3 * 2 * BINS
coords = np.arange(2, CROP - 2)
in_block = (coords % 8 >= 1) & (coords % 8 <= 6)
triple_in_block = (coords[:-2] % 8 >= 1) & (coords[:-2] % 8 <= 4)
horizontal_mask = in_block[:, None] & triple_in_block[None, :]
vertical_mask = horizontal_mask.T


class CooccurrenceCNN(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant != VARIANT:
            raise ValueError(f"Unknown residual variant {variant}")
        self.features = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(), nn.Linear(48 * 9, 2),
        )

    def forward(self, views):
        mapped = views.flatten(0, 1).float().contiguous(memory_format=torch.channels_last)
        return self.features(mapped).reshape(len(views), views.shape[1], 2).mean(dim=1)


def triple_histogram(codes, horizontal):
    if horizontal:
        triples = BINS ** 2 * codes[..., :-2] + BINS * codes[..., 1:-1] + codes[..., 2:]
        kept = triples[..., horizontal_mask]
    else:
        triples = BINS ** 2 * codes[..., :-2, :] + BINS * codes[..., 1:-1, :] + codes[..., 2:, :]
        kept = triples[..., vertical_mask]
    leading = codes.shape[:-2]
    batch_count = int(np.prod(leading))
    offsets = (np.arange(batch_count, dtype=np.int64) * BINS ** 3).reshape(*leading, 1)
    counts = np.bincount((kept + offsets).ravel(), minlength=batch_count * BINS ** 3)
    return counts.reshape(*leading, BINS, BINS, BINS)


def quantize(residual, step):
    midpoint = BINS // 2
    return np.clip(np.rint(residual / step), -midpoint, midpoint).astype(np.int16) + midpoint


def descriptors(patches):
    rgb = patches.astype(np.int32)
    red, green, blue = (rgb[:, i] for i in range(3))
    luminance = (77 * red + 150 * green + 29 * blue + 128) // 256
    planes = np.stack((luminance, red - green, blue - green), axis=1)
    center = planes[..., 2:-2, 2:-2]
    left, right = planes[..., 2:-2, 1:-3], planes[..., 2:-2, 3:-1]
    above, below = planes[..., 1:-3, 2:-2], planes[..., 3:-1, 2:-2]
    mixed = center - left - above + planes[..., 1:-3, 1:-3]
    families = ((center - left, center - above, 2.0),
                (left + right - 2 * center, above + below - 2 * center, 4.0),
                (mixed, mixed, 2.0))
    features = []
    for horizontal, vertical, fixed in families:
        scale = np.maximum(1.0, (np.abs(horizontal).mean(axis=(-2, -1), keepdims=True)
                                + np.abs(vertical).mean(axis=(-2, -1), keepdims=True)) * 0.5)
        for step in (fixed, scale):
            counts = triple_histogram(quantize(horizontal, step), horizontal=True)
            counts += triple_histogram(quantize(vertical, step), horizontal=False)
            symmetric = counts + counts[..., ::-1, ::-1, ::-1]
            symmetric = symmetric + symmetric.swapaxes(-3, -1)
            probability = symmetric / symmetric.sum(axis=(-3, -2, -1), keepdims=True)
            features.append(np.log1p(1000 * probability).reshape(len(patches), -1, BINS, BINS))
    return np.concatenate(features, axis=1).astype(np.float16)


def prepare(sample):
    size, patches = prepare_image(sample, QUALITY)
    return size, descriptors(patches)


def cache(samples, deadline):
    print(f"Preparing residual statistics for {len(samples)} labeled or inference images", flush=True)
    values = np.empty((len(samples), 5, CHANNELS, BINS, BINS), dtype=np.float16)
    sizes = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(samples), 32):
            check_time(deadline)
            for i, (size, feature) in enumerate(executor.map(prepare, samples[start:start + 32]), start=start):
                values[i] = feature
                sizes.append(size)
            done = min(start + 32, len(samples))
            if done % 1024 == 0 or done == len(samples):
                print(f"Prepared {done}/{len(samples)}", flush=True)
    return torch.from_numpy(values), sizes


def run(args, device, deadline):
    data = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    split, identity = labeled_splits(data, {"train_per_class": 3200, "val_per_class": 800,
                                           "short_side": CROP, "crop_size": CROP})
    samples = split["train"] + split["val"]
    train_end = len(split["train"])
    labels = torch.tensor([sample.label for sample in samples])
    report = {"epochs": SCREEN_EPOCHS, "seeds": SEEDS, "quality": QUALITY,
              "split_sha256": identity["split_sha256"], "results": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("prepare_labeled")
    values, sizes = cache(samples, deadline)
    rgb, _ = rgb_cache(split["val"], QUALITY, deadline)
    real = sorted((i for i in range(train_end, len(samples)) if samples[i].label == 0),
                  key=lambda i: sha256(f"threshold-audit:{samples[i].image_id}".encode()).digest())
    offset = 0

    def train(folder, rows):
        return [fit(folder / f"s{seed}", VARIANT, seed, SCREEN_EPOCHS, values, labels, rows, device,
                    deadline, model_factory=CooccurrenceCNN) for seed in SEEDS]

    for cohort in SCREEN:
        record(f"screen_{cohort}")
        train_rows, selected = sample_rows(samples, sizes, train_end, cohort)
        fake = selected[len(selected) // 2:].tolist()
        rows = torch.tensor(real[offset:offset + len(fake)] + fake)
        if len(rows) != 2 * len(fake):
            raise ValueError("Insufficient disjoint real validation samples")
        offset += len(fake)
        folder = (args.group_reference / f"confirm_{cohort}_reference" if cohort == 256
                  else args.night_reference / f"screen_{cohort}_{ARCHITECTURE}")
        reference = restore_ensemble([folder / f"s{seed}" / "last.pt" for seed in SEEDS], device)
        model = train(args.output / f"cooccurrence_{cohort}", train_rows)
        truth = labels[rows].numpy()
        before = scores(reference, rgb, rows - train_end, device, deadline)
        after = scores(model, values, rows, device, deadline)
        a = binary_metrics(truth, before > 0.5, before)
        b = binary_metrics(truth, after > 0.5, after)
        report["results"][str(cohort)] = {"reference": a, "cooccurrence": b,
                                          "paired_accuracy": accuracy_interval(truth, before, after)}
        print(f"{cohort}: accuracy {a['accuracy']:.3%} -> {b['accuracy']:.3%}, "
              f"AUROC {a['auroc']:.4f} -> {b['auroc']:.4f}", flush=True)

    record("full_training")
    train_rows, val_rows = sample_rows(samples, sizes, train_end)
    models = train(args.output / "full", train_rows)
    probability = scores(models, values, val_rows, device, deadline)
    report["full_validation"] = binary_metrics(labels[val_rows].tolist(), probability > 0.5, probability)
    report["train_count"] = len(train_rows)
    del values, rgb

    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"),
                     Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    encoded, _ = cache(test, deadline)
    predicted = scores(models, encoded, torch.arange(len(test)), device, deadline) > 0.5
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(label)) for sample, label in zip(test, predicted, strict=True)))
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

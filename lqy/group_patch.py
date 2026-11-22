"""Compare size-balanced patch CNN training with and without JPEG consistency."""

import argparse
import time
from bisect import bisect_right
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from detection.data import read_test
from detection.metrics import binary_metrics
from detection.training import load_checkpoint, write_csv, write_json

from .cnn import device_for, run_lock
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, PatchCNN, cache, fit, sample_rows, scores
from .robust import labeled_splits


ARCHITECTURE = "rgb_and_residual"
RECIPES = ("balanced", "balanced_consistency")
SCREEN = (512, 1024)
CONFIRM = 256
QUALITIES = (75, 90)


def group_weights(rows, labels, sizes):
    indices = rows.tolist()
    real_sides = [min(sizes[i]) for i in indices if labels[i] == 0]
    quartiles = sorted(set(np.quantile(real_sides, (0.25, 0.5, 0.75)).tolist()))
    groups = [(0, bisect_right(quartiles, min(sizes[i]))) if labels[i] == 0
              else (1, sizes[i]) for i in indices]
    counts = Counter(groups)
    counts_per_class = Counter(label for label, _ in counts)
    return torch.tensor([1 / (counts[group] * counts_per_class[group[0]]) for group in groups],
                        dtype=torch.float64)


def fit_ensemble(output, recipe, q75, q90, labels, sizes, rows, device, deadline):
    weights = None if recipe == "reference" else group_weights(rows, labels, sizes)
    paired = q90 if recipe == "balanced_consistency" else None
    return [fit(output / f"s{seed}", ARCHITECTURE, seed, SCREEN_EPOCHS, q75, labels, rows, device,
                deadline, sampling_weights=weights, paired_data=paired) for seed in SEEDS]


def restore_ensemble(paths, device):
    models = []
    for seed, path in zip(SEEDS, paths, strict=True):
        state = load_checkpoint(path)
        if (state["variant"], state["seed"], state["epoch"]) != (ARCHITECTURE, seed, SCREEN_EPOCHS):
            raise ValueError(f"Incompatible reference checkpoint {path}")
        model = PatchCNN(ARCHITECTURE).to(device, memory_format=torch.channels_last)
        model.load_state_dict(state["weights"])
        models.append(model)
    return models


def score_views(models, q75, q90, rows, device, deadline):
    first = scores(models, q75, rows, device, deadline)
    second = scores(models, q90, rows, device, deadline)
    return {"q75": first, "q90": second, "mean": (first + second) * 0.5}


def measure(models, q75, q90, rows, labels, device, deadline):
    truth = labels[rows].tolist()
    values = score_views(models, q75, q90, rows, device, deadline)
    return {key: binary_metrics(truth, value > 0.5, value) for key, value in values.items()}


def print_metrics(name, metrics):
    display = "  ".join(f"{profile}={result['accuracy']:.3%}" for profile, result in metrics.items())
    print(f"{name}: {display}", flush=True)


def run(args, device, deadline):
    data_args = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    splits, identity = labeled_splits(data_args, {"train_per_class": 3200, "val_per_class": 800,
                                                   "short_side": CROP, "crop_size": CROP})
    samples = splits["train"] + splits["val"]
    labels = torch.tensor([sample.label for sample in samples])
    train_end = len(splits["train"])
    report = {"initialization": "random", "seeds": SEEDS, "epochs": SCREEN_EPOCHS,
              "split_sha256": identity["split_sha256"], "screen_cohorts": SCREEN,
              "confirmation_cohort": CONFIRM, "jpeg_qualities": QUALITIES, "screen": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("prepare_labeled")
    q75, sizes = cache(samples, QUALITIES[0], deadline)
    q90, sizes90 = cache(samples, QUALITIES[1], deadline)
    if sizes != sizes90:
        raise ValueError("Labeled views are misaligned")

    record("screening")
    for cohort in SCREEN:
        train_rows, val_rows = sample_rows(samples, sizes, train_end, cohort)
        results = {}
        report["screen"][str(cohort)] = results
        reference_root = args.reference / f"screen_{cohort}_{ARCHITECTURE}"
        reference = restore_ensemble([reference_root / f"s{seed}" / "last.pt" for seed in SEEDS], device)
        results["reference"] = measure(reference, q75, q90, val_rows, labels, device, deadline)
        print_metrics(f"Screen {cohort} reference", results["reference"])
        for recipe in RECIPES:
            models = fit_ensemble(args.output / f"screen_{cohort}_{recipe}", recipe, q75, q90,
                                  labels, sizes, train_rows, device, deadline)
            results[recipe] = measure(models, q75, q90, val_rows, labels, device, deadline)
            print_metrics(f"Screen {cohort} {recipe}", results[recipe])
            record("screening")

    def rank(recipe):
        outcomes = [item[recipe] for item in report["screen"].values()]
        return (min(item["mean"]["accuracy"] for item in outcomes),
                min(item[view]["accuracy"] for item in outcomes for view in ("q75", "q90")),
                sum(item["mean"]["accuracy"] for item in outcomes))

    winner = max(RECIPES, key=rank)
    report["selected_recipe"] = winner
    report["confirmation"] = {}
    record("confirmation")
    train_rows, val_rows = sample_rows(samples, sizes, train_end, CONFIRM)
    for recipe in ("reference", winner):
        models = fit_ensemble(args.output / f"confirm_{CONFIRM}_{recipe}", recipe, q75, q90,
                              labels, sizes, train_rows, device, deadline)
        metric = measure(models, q75, q90, val_rows, labels, device, deadline)
        report["confirmation"][recipe] = metric
        print_metrics(f"Confirm {CONFIRM} {recipe}", metric)
        record("confirmation")

    record("full_training")
    train_rows, val_rows = sample_rows(samples, sizes, train_end)
    models = fit_ensemble(args.output / f"full_{winner}", winner, q75, q90, labels, sizes,
                          train_rows, device, deadline)
    report["full_train_count"] = len(train_rows)
    report["full_validation"] = measure(models, q75, q90, val_rows, labels, device, deadline)
    print_metrics("Full training", report["full_validation"])
    del q75, q90

    record("export")
    template = Path("ai-generated-image-detection-benchmark/sample_submission.csv")
    test = read_test(template, Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError(f"Expected 1,000 test images, got {len(test)}")
    first, test_sizes = cache(test, QUALITIES[0], deadline)
    second, sizes90 = cache(test, QUALITIES[1], deadline)
    if test_sizes != sizes90:
        raise ValueError("Test views are misaligned")
    values = score_views(models, first, second, torch.arange(len(test)), device, deadline)["mean"]
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(value > 0.5)) for sample, value in zip(test, values, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 1800)


if __name__ == "__main__":
    main()

"""Test whether same-size training support improves an unseen labeled subgroup."""

import argparse
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from detection.training import write_json

from .cnn import device_for, read_json, run_lock
from .codec_anchor import cutoff, exclude_anchor, prepare_data, recenter, view_pair
from .codec_response import train_variant
from .codec_scale import HYBRID, QUANTILE, HybridResponseCNN, restore_candidate
from .deep_fusion import comparison, spectral_views
from .group_patch import ARCHITECTURE
from .night_train import SEEDS, SCREEN_EPOCHS, fit, scores
from .spectrum_patch import VARIANT as SPECTRUM_VARIANT, SpectrumPatchCNN


def tagged(sample):
    with Image.open(sample.path) as image:
        return "matplotlib" in str(image.info.get("Software", "")).casefold()


def subgroup_recall(truth, original, current):
    correct_before = (original > 0.5) == truth
    correct_after = (current > 0.5) == truth
    change = correct_after.astype(np.int8) - correct_before.astype(np.int8)
    rng = np.random.default_rng(17)
    random_rows = rng.integers(len(change), size=(3000, len(change)))
    return {"n": len(change), "before": float(correct_before.mean()), "after": float(correct_after.mean()),
            "gain": float(change.mean()),
            "paired_95_percent": np.quantile(change[random_rows].mean(axis=1), (0.025, 0.975)).tolist()}


def benchmark(args, native, candidate_views, labels, training, anchor, validation, groups, device, deadline):
    folder = args.output / "matched_reference"
    spatial = [fit(folder / "spatial" / f"s{seed}", ARCHITECTURE, seed, SCREEN_EPOCHS, native, labels,
                   training, device, deadline) for seed in SEEDS]
    spectrum = spectral_views(native, torch.arange(len(labels)), deadline)
    frequency = [fit(folder / "spectrum" / f"s{seed}", SPECTRUM_VARIANT, seed, SCREEN_EPOCHS, spectrum,
                     labels, training, device, deadline, model_factory=SpectrumPatchCNN) for seed in SEEDS]

    def reference(rows):
        return (scores(spatial, native, rows, device, deadline) + scores(frequency, spectrum, rows, device, deadline)) / 2

    models = restore_candidate(args.output / "same_size" / HYBRID, HYBRID, HybridResponseCNN, device)
    threshold = cutoff(scores(models, candidate_views, anchor, device, deadline), QUANTILE)
    predicted = recenter(scores(models, candidate_views, validation, device, deadline), threshold)
    original = reference(validation)
    original_threshold = cutoff(reference(anchor), QUANTILE)
    calibrated_reference = recenter(original, original_threshold)
    truth = labels[validation].numpy()
    masks = {str(group): groups == group for group in np.unique(groups)}
    report = {"stage": "complete", "train_count_both": len(training),
              "reference_default_threshold": comparison(truth, original, predicted),
              "reference_calibrated": comparison(truth, calibrated_reference, predicted),
              "subgroups_vs_calibrated_reference": {
                  group: subgroup_recall(truth[mask], calibrated_reference[mask], predicted[mask])
                  for group, mask in masks.items()}}
    write_json(args.output / "benchmark.json", report)
    for name in ("reference_default_threshold", "reference_calibrated"):
        result = report[name]
        print(f"{name}: accuracy {result['reference']['accuracy']:.3%} -> {result['deep_fusion']['accuracy']:.3%}", flush=True)


def run(args, device, deadline):
    samples, labels, sizes, anchor, folds, full, table, identity = prepare_data((512, 1024, 128, 256))
    previous = read_json(args.hybrid_reference / "summary.json")
    if previous["split_sha256"] != identity or previous["variant"] != HYBRID:
        raise ValueError("Reference checkpoint metadata does not match this experiment")
    previous_train, val = folds[256]
    previous_train = exclude_anchor(previous_train, labels, anchor)
    real_train = previous_train[labels[previous_train] == 0]
    fake_train = previous_train[labels[previous_train] == 1]
    support = torch.tensor([i for i in full.tolist() if samples[i].label == 1
                            and sizes[i] == (256, 256) and not tagged(samples[i])])
    if not 0 < len(support) <= len(fake_train):
        raise ValueError("Expected a smaller, nonempty set of same-size training examples")
    kept = sorted(fake_train.tolist(),
                  key=lambda i: sha256(f"same-size-support:{samples[i].image_id}".encode()).digest())
    replace = torch.tensor(kept[:len(fake_train) - len(support)], dtype=torch.long)
    train = torch.cat((real_train, replace, support))
    if torch.isin(train, anchor).any() or torch.isin(train, val).any() or len(train) != len(previous_train):
        raise ValueError("Training counts or disjoint holdouts changed")
    report = {"stage": "training", "split_sha256": identity, "train_count_both": len(train),
              "replaced_with_unmarked_256": len(support), "photo_calibration_quantile": QUANTILE}
    if args.benchmark:
        completed = read_json(args.output / "summary.json")
        if completed["stage"] != "complete" or completed["split_sha256"] != identity or completed["train_count_both"] != len(train):
            raise ValueError("The equal-support candidate must finish on the same labeled split first")
        print(f"Training the reference models on the same {len(train)} images", flush=True)
    else:
        write_json(args.output / "summary.json", report)
        print(f"Keeping {len(train)} training rows and replacing {len(support)} generated examples", flush=True)
    rows = torch.cat((train, anchor, val))
    native, inputs = view_pair([samples[i] for i in rows.tolist()], table, deadline)
    train_rows = torch.arange(len(train))
    anchor_rows = torch.arange(len(train), len(train) + len(anchor))
    val_rows = torch.arange(len(train) + len(anchor), len(rows))
    groups = np.asarray(["real" if samples[i].label == 0 else "marked_ai" if tagged(samples[i]) else "unmarked_ai"
                         for i in val.tolist()])
    if args.benchmark:
        benchmark(args, native, inputs, labels[rows], train_rows, anchor_rows, val_rows, groups, device, deadline)
        return
    models = train_variant(args.output / "same_size" / HYBRID, HYBRID, inputs, labels[rows], sizes, train_rows,
                           device, deadline, model_factory=HybridResponseCNN)
    original = restore_candidate(args.hybrid_reference / "group-256" / HYBRID, HYBRID, HybridResponseCNN, device)

    def calibrated(model):
        threshold = cutoff(scores(model, inputs, anchor_rows, device, deadline), QUANTILE)
        return recenter(scores(model, inputs, val_rows, device, deadline), threshold)

    before, after = calibrated(original), calibrated(models)
    truth = labels[val].numpy()
    report["accuracy_comparison"] = comparison(truth, before, after)
    masks = {str(group): groups == group for group in np.unique(groups)}
    report["subgroups"] = {group: subgroup_recall(truth[use], before[use], after[use]) for group, use in masks.items()}
    report["stage"] = "complete"
    write_json(args.output / "summary.json", report)
    for group, result in report["subgroups"].items():
        print(f"{group}: {result['before']:.3%} -> {result['after']:.3%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--benchmark", action="store_true", help="Train the original algorithm on the exact support rows")
    parser.add_argument("--hybrid-reference", type=Path, default=Path("lqy/runs/codec-scale/hybrid-response"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

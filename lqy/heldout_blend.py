"""Check whether existing scratch-trained models complement each other out of group."""

import argparse
import itertools
import time
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from detection.data import read_test
from detection.training import load_checkpoint, write_csv, write_json

from .cnn import device_for, run_lock
from .deep_fusion import MIN_CONFIRM_GAIN, MIN_SCREEN_GAIN, bootstrap_mean_gain, comparison
from .group_patch import ARCHITECTURE, restore_ensemble
from .local_global import VARIANT as GLOBAL_VARIANT, LocalGlobalCNN, cache as global_cache
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, scores
from .residual_cooccurrence import VARIANT as COOCCURRENCE_VARIANT, CooccurrenceCNN, cache as cooccurrence_cache
from .robust import labeled_splits
from .spectrum_patch import VARIANT as SPECTRUM_VARIANT, SpectrumPatchCNN, cache as spectrum_cache


COHORTS = (512, 1024, 256)
NAMES = ("spatial_spectrum", "jpeg_history", "cooccurrence", "local_global")
WEIGHTS = np.asarray([(10 - sum(tail), *tail) for tail in itertools.product(range(7), repeat=3)
                      if sum(tail) <= 6], dtype=np.float64) / 10


def restore(folder, constructor, variant, device):
    result = []
    for seed in SEEDS:
        path = folder / f"s{seed}" / "last.pt"
        checkpoint = load_checkpoint(path)
        if (checkpoint["variant"], checkpoint["seed"], checkpoint["epoch"]) != (variant, seed, SCREEN_EPOCHS):
            raise ValueError(f"Incompatible model checkpoint: {path}")
        model = constructor(variant).to(device, memory_format=torch.channels_last)
        model.load_state_dict(checkpoint["weights"])
        result.append(model)
    return result


def inputs(samples, deadline):
    spectral, sizes = spectrum_cache(samples, deadline)
    joint, _ = global_cache(samples, deadline)
    residual, _ = cooccurrence_cache(samples, deadline)
    return (spectral, joint, residual), sizes


def probabilities(cohort, packed, rows, device, deadline):
    spectral, joint, residual = packed
    runs = Path("lqy/runs")
    night = runs / "night-train/2026-11-21"
    if cohort == 256:
        raw_root = runs / "group-patch/round-01/confirm_256_reference"
    elif cohort is None:
        raw_root = night / "online-probe"
    else:
        raw_root = night / f"screen_{cohort}_{ARCHITECTURE}"
    spec_root = runs / "spectrum-patch/round-01" / ("online-probe" if cohort is None else f"spectrum_{cohort}")
    jpeg_root = runs / "jpeg-history/round-01" / ("full_matched" if cohort is None else f"matched_{cohort}")
    global_root = runs / "local-global/round-01" / ("full" if cohort is None else f"local_global_{cohort}")
    residual_root = runs / "residual-cooc/round-01" / ("full" if cohort is None else f"cooccurrence_{cohort}")

    def raw(path):
        return restore_ensemble([path / f"s{seed}" / "last.pt" for seed in SEEDS], device)

    local = spectral[:, :, :3]
    spatial_score = scores(raw(raw_root), local, rows, device, deadline)
    spectral_score = scores(restore(spec_root, SpectrumPatchCNN, SPECTRUM_VARIANT, device), spectral, rows, device, deadline)
    jpeg_score = scores(raw(jpeg_root), local, rows, device, deadline)
    residual_score = scores(restore(residual_root, CooccurrenceCNN, COOCCURRENCE_VARIANT, device), residual, rows, device, deadline)
    global_score = scores(restore(global_root, LocalGlobalCNN, GLOBAL_VARIANT, device), joint, rows, device, deadline)
    return np.stack(((spatial_score + spectral_score) * 0.5, jpeg_score, residual_score, global_score), axis=1)


def choose(groups):
    def rank(weight):
        accuracy, squared_error = [], []
        for truth, matrix in groups:
            p = matrix @ weight
            accuracy.append(np.mean((p > 0.5) == truth))
            squared_error.append(np.mean((p - truth) ** 2))
        return min(accuracy), np.mean(accuracy), -np.mean(squared_error), weight[0]

    return max(WEIGHTS, key=rank)


def run(args, device, deadline):
    source = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    splits, identity = labeled_splits(source, {"train_per_class": 3200, "val_per_class": 800,
                                               "short_side": CROP, "crop_size": CROP})
    sample = splits["val"]
    packed, sizes = inputs(sample, deadline)
    groups = {}
    ordered = sorted(range(len(sample)), key=lambda i: sha256(f"threshold-audit:{sample[i].image_id}".encode()).digest())
    real = [i for i in ordered if sample[i].label == 0]
    offset = 0
    for cohort in COHORTS:
        fake = [i for i in ordered if sample[i].label == 1 and sizes[i] == (cohort, cohort)]
        rows = torch.tensor(real[offset:offset + len(fake)] + fake)
        offset += len(fake)
        if len(rows) != 2 * len(fake):
            raise ValueError("Insufficient disjoint real holdout samples")
        truth = np.array([sample[i].label for i in rows.tolist()])
        groups[cohort] = truth, probabilities(cohort, packed, rows, device, deadline)
        print(f"Scored held-out cohort {cohort}", flush=True)
    del packed

    results, pairs = {}, []
    for cohort, fit_on in ((512, 1024), (1024, 512)):
        weight = choose([groups[fit_on]])
        truth, matrix = groups[cohort]
        result = matrix @ weight
        results[str(cohort)] = {"weights": weight.tolist(), "fit_on": fit_on,
                                **comparison(truth, matrix[:, 0], result)}
        pairs.append((truth, matrix[:, 0], result))
    aggregate = bootstrap_mean_gain(pairs)
    weight = choose([groups[512], groups[1024]])
    truth, independent = groups[256]
    confirm = comparison(truth, independent[:, 0], independent @ weight)
    screen_passes = (all(all(delta > 0 for delta in item["delta"].values()) for item in results.values())
                     and aggregate["mean_gain"] >= MIN_SCREEN_GAIN and aggregate["bootstrap_95_percent"][0] > 0)
    accepted = screen_passes and confirm["delta"]["accuracy"] >= MIN_CONFIRM_GAIN and confirm["delta"]["auroc"] > 0
    report = {"stage": "stopped_at_screen" if not screen_passes else "stopped_at_confirmation",
              "methods": NAMES, "split_sha256": identity["split_sha256"], "screen": results,
              "screen_paired_summary": aggregate, "confirmation": confirm, "selected_weights": weight.tolist()}
    print(f"Cross-cohort accuracy gain: {aggregate}; confirmation delta: {confirm['delta']}", flush=True)
    if accepted:
        test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
        if len(test) != 1000:
            raise ValueError("Expected 1,000 test images")
        data, _ = inputs(test, deadline)
        predicted = probabilities(None, data, torch.arange(len(test)), device, deadline) @ weight > 0.5
        write_csv(args.output / "submission.csv", ("image_id", "label"),
                  zip((picture.image_id for picture in test), predicted.astype(int), strict=True))
        report["stage"] = "complete"
    write_json(args.output / "summary.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="cpu")
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 4), time.monotonic() + 2400)


if __name__ == "__main__":
    main()

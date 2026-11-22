"""Check real-only spectral models and training checkpoints before submission."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from detection.data import read_test
from detection.training import load_checkpoint, write_csv, write_json

from .cnn import device_for, run_lock
from .deep_fusion import CONFIRM, MIN_CONFIRM_GAIN, MIN_SCREEN_GAIN, SCREEN, bootstrap_mean_gain, combined_reference, comparison
from .group_patch import ARCHITECTURE
from .night_train import BATCH, CROP, SEEDS, PatchCNN, check_time, fit, sample_rows, scores
from .robust import labeled_splits
from .spectrum_patch import QUALITY, VARIANT, SpectrumPatchCNN, cache


TRAJECTORIES = {"epoch_3": (3,), "epoch_9": (9,), "epoch_12": (12,), "averaged_6_9_12": (6, 9, 12)}
REAL_ONLY = ("masked_real_spectrum", "real_spectrum_distance")


def spectral_summary(packed):
    parts = []
    for block in packed.split(48):
        spectra = block[:, :, 3:].numpy().astype(np.float32) / 255 - 0.5
        tiles = spectra.reshape(len(block), 5, 3, 4, CROP // 4, 4, CROP // 4)
        means = tiles.mean(axis=(4, 6))
        energy = np.sqrt((tiles ** 2).mean(axis=(4, 6)))
        summary = np.concatenate([x.reshape(len(block), -1) for x in
                                  (means.mean(axis=1), means.std(axis=1),
                                   energy.mean(axis=1), energy.std(axis=1))], axis=1)
        parts.append(summary)
    return np.concatenate(parts)


def train_real_model(features, labels, training):
    real = training[labels[training] == 0]
    fake = training[labels[training] == 1]
    if len(real) != len(fake) or len(real) < 100:
        raise ValueError("Real-only calibration requires a balanced labeled training partition")
    rng = np.random.default_rng(17)
    real, fake = rng.permutation(real), rng.permutation(fake)
    reserve = len(real) // 5
    values = features[real[reserve:]].astype(np.float64)
    center = values.mean(axis=0)
    scale = values.std(axis=0).clip(min=0.012)
    normalized = (values - center) / scale
    covariance = normalized.T @ normalized / (len(normalized) - 1)
    covariance += 0.10 * np.eye(len(center))
    cholesky = np.linalg.cholesky(covariance)
    dimension = np.arange(len(center))
    masks = []
    for target in np.array_split(rng.permutation(dimension), 5):
        seen = np.setdiff1d(dimension, target)
        weights = np.linalg.solve(covariance[np.ix_(seen, seen)], covariance[np.ix_(seen, target)])
        residual = normalized[:, target] - normalized[:, seen] @ weights
        uncertainty = np.mean(residual ** 2, axis=0).clip(min=0.06)
        masks.append((seen, target, weights, uncertainty))

    def anomaly(rows):
        vector = (rows - center) / scale
        whitened = np.linalg.solve(cholesky, vector.T).T
        masked_error = np.zeros(len(rows))
        for seen, target, matrix, var in masks:
            error = vector[:, target] - vector[:, seen] @ matrix
            masked_error += ((error ** 2) / var).sum(axis=1)
        return {REAL_ONLY[0]: masked_error / len(center), REAL_ONLY[1]: (whitened ** 2).mean(axis=1)}

    calibration = np.concatenate((real[:reserve], fake[:reserve]))
    truth = labels[calibration]
    limits = {}
    for name, value in anomaly(features[calibration]).items():
        thresholds = np.unique(value)
        median = float(np.median(value))
        selected = float(max(thresholds, key=lambda t: (np.mean((value > t) == truth), -abs(t - median))))
        width = max(float(np.quantile(value, 0.75) - np.quantile(value, 0.25)), 1e-6)
        limits[name] = (selected, width)

    def predict(rows):
        output = {}
        for name, value in anomaly(rows).items():
            threshold, spread = limits[name]
            output[name] = 1 / (1 + np.exp(-np.clip((value - threshold) / spread, -32, 32)))
        return output

    return predict, {"real_model_fit": len(values), "calibration": len(calibration), "thresholds": limits}


@torch.no_grad()
def update_batch_norm(model, data, training, device, deadline):
    model.eval()
    normalizations = [part for part in model.modules() if isinstance(part, nn.BatchNorm2d)]
    for part in normalizations:
        part.reset_running_stats()
        part.momentum = None
        part.train()
    generator = torch.Generator().manual_seed(42)
    rows = training[torch.randperm(len(training), generator=generator)[:20 * BATCH]]
    for batch in rows.split(BATCH):
        check_time(deadline)
        model(data[batch].to(device))
    model.eval()


def restore_weights(paths, constructor, variant, data, train, device, deadline):
    states = [load_checkpoint(path)["weights"] for path in paths]
    weights = states[-1]
    if len(states) > 1:
        weights = OrderedDict((key, torch.stack([item[key] for item in states]).mean(dim=0)
                               if value.is_floating_point() else value) for key, value in weights.items())
    model = constructor(variant).to(device, memory_format=torch.channels_last)
    model.load_state_dict(weights)
    if len(states) > 1:
        update_batch_norm(model, data, train, device, deadline)
    return model.eval()


def train_trajectories(folder, local, spectrum, labels, training, device, deadline, variants):
    wanted = sorted({epoch for path in variants.values() for epoch in path})
    sources = (("spatial", PatchCNN, ARCHITECTURE, local), ("spectrum", SpectrumPatchCNN, VARIANT, spectrum))
    result = {name: [] for name in variants}
    for branch, constructor, variant, data in sources:
        seed_paths = []
        for seed in SEEDS:
            output = folder / branch / f"s{seed}"
            fit(output, variant, seed, max(wanted), data, labels, training, device, deadline,
                model_factory=constructor, snapshot_epochs=wanted)
            seed_paths.append(output)
        for name, epochs in variants.items():
            models = [restore_weights([root / f"epoch-{epoch}.pt" for epoch in epochs], constructor,
                                      variant, data, training, device, deadline) for root in seed_paths]
            result[name].append((models, data))
    return result


def trajectory_scores(branches, rows, device, deadline):
    return sum(scores(models, data, rows, device, deadline) for models, data in branches) / len(branches)


def gate(results, paired):
    aggregate = bootstrap_mean_gain(paired)
    positive = all(all(delta > 0 for delta in result["delta"].values()) for result in results.values())
    return {**aggregate, "passes": (positive and aggregate["mean_gain"] >= MIN_SCREEN_GAIN
                                    and aggregate["bootstrap_95_percent"][0] > 0)}


def run(args, device, deadline):
    sources = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    splits, identity = labeled_splits(sources, {"train_per_class": 3200, "val_per_class": 800,
                                                "short_side": CROP, "crop_size": CROP})
    samples = splits["train"] + splits["val"]
    train_count = len(splits["train"])
    truth = torch.tensor([sample.label for sample in samples])
    report = {"split_sha256": identity["split_sha256"], "reference": "56.5% spatial/spectrum ensemble",
              "trajectories": TRAJECTORIES, "real_only": REAL_ONLY, "results": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("prepare_labeled")
    spectrum, sizes = cache(samples, deadline)
    local = spectrum[:, :, :3]
    compact = spectral_summary(spectrum)
    real = sorted((i for i in range(train_count, len(samples)) if samples[i].label == 0),
                  key=lambda i: sha256(f"threshold-audit:{samples[i].image_id}".encode()).digest())
    paired = {name: [] for name in (*REAL_ONLY, *TRAJECTORIES)}
    offset = 0

    def fold_rows(cohort):
        nonlocal offset
        train, group = sample_rows(samples, sizes, train_count, cohort)
        fake = group[len(group) // 2:].tolist()
        val = torch.tensor(real[offset:offset + len(fake)] + fake)
        offset += len(fake)
        if len(val) != 2 * len(fake):
            raise ValueError("Insufficient disjoint real validation images")
        return train, val

    def compare_candidates(cohort, folder):
        train, val = fold_rows(cohort)
        old = combined_reference(args, cohort, local, truth, train, val, device, deadline)
        with ThreadPoolExecutor(max_workers=1) as executor:
            real_job = executor.submit(train_real_model, compact, truth.numpy(), train.numpy())
            branches = train_trajectories(folder, local, spectrum, truth, train, device, deadline, TRAJECTORIES)
            candidates = {name: trajectory_scores(branch, val, device, deadline) for name, branch in branches.items()}
            predict, calibration = real_job.result()
            candidates.update(predict(compact[val.numpy()]))
        group_truth = truth[val].numpy()
        result = {name: comparison(group_truth, old, value) for name, value in candidates.items()}
        for name, value in candidates.items():
            paired[name].append((group_truth, old, value))
            print(f"{cohort} {name}: accuracy {result[name]['deep_fusion']['accuracy']:.3%}, "
                  f"AUROC {result[name]['deep_fusion']['auroc']:.4f}", flush=True)
        return {"calibration": calibration, "candidates": result}

    for cohort in SCREEN:
        record(f"screen_{cohort}")
        report["results"][str(cohort)] = compare_candidates(cohort, args.output / f"cohort_{cohort}")
    gates = {name: gate({cohort: item["candidates"][name] for cohort, item in report["results"].items()}, pairs)
             for name, pairs in paired.items()}
    report["gates"] = gates
    ranked = sorted((name for name in gates if gates[name]["passes"]),
                    key=lambda name: gates[name]["mean_gain"], reverse=True)
    if not ranked:
        record("stopped_at_screen")
        print(f"All candidates reached the screening decision: {gates}", flush=True)
        return

    record("confirm_128")
    train, val = fold_rows(CONFIRM)
    folder = args.output / "confirmation"
    control = train_trajectories(folder / "reference", local, spectrum, truth, train, device, deadline, {"reference": (6,)})
    baseline = trajectory_scores(control["reference"], val, device, deadline)
    report["confirmation"] = {}
    selected = None
    for name in ranked:
        if name in REAL_ONLY:
            predict, _ = train_real_model(compact, truth.numpy(), train.numpy())
            result = predict(compact[val.numpy()])[name]
        else:
            model = train_trajectories(folder / name, local, spectrum, truth, train, device, deadline,
                                       {name: TRAJECTORIES[name]})
            result = trajectory_scores(model[name], val, device, deadline)
        measured = comparison(truth[val].numpy(), baseline, result)
        report["confirmation"][name] = measured
        if measured["delta"]["accuracy"] >= MIN_CONFIRM_GAIN and measured["delta"]["auroc"] > 0:
            selected = name
            break
    if selected is None:
        record("stopped_at_confirmation")
        return

    report["selected"] = selected
    record("full_training")
    train, _ = sample_rows(samples, sizes, train_count)
    if selected in REAL_ONLY:
        predict, calibration = train_real_model(compact, truth.numpy(), train.numpy())
        report["full_calibration"] = calibration
    else:
        full = train_trajectories(args.output / "full", local, spectrum, truth, train, device, deadline,
                                  {selected: TRAJECTORIES[selected]})[selected]
        full = [models for models, _ in full]
    del spectrum, local
    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"),
                     Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    inputs, _ = cache(test, deadline)
    if selected in REAL_ONLY:
        probability = predict(spectral_summary(inputs))[selected]
    else:
        probability = trajectory_scores(list(zip(full, (inputs[:, :, :3], inputs), strict=True)),
                                        torch.arange(len(test)), device, deadline)
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(score > 0.5)) for sample, score in zip(test, probability, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 5400)


if __name__ == "__main__":
    main()

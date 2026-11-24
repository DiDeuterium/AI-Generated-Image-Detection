"""Replay the frozen 0.628 release and test a single, retrained branch replacement."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from detection.data import read_test
from detection.training import load_checkpoint
from lqy.canonical_augmentation import CORRELATION_VARIANT, CorrelationCNN
from lqy.codec_anchor import prepare_data, recenter
from lqy.codec_canonical import compare_views, validation_halves
from lqy.codec_final import profile_gain
from lqy.codec_scale import restore_candidate
from lqy.night_train import SEEDS, SCREEN_EPOCHS, scores
from lqy.source_holdout import ReleasedReference

from lqy_resize_retrain import TEMPLATE, TEST_DIR
from ssd.engine import compute_metrics

BUNDLE = Path("fqm/runs/lqy_replay_assets_20261122")
NEW = Path("fqm/runs/lqy_resize_20261122")
PROFILES = ("native", "jpeg76", "alternate_resize_jpeg", "webp")


def read_json(path):
    return json.loads(path.read_text())


def branch(paths, device):
    models = []
    for seed, path in zip(SEEDS, paths, strict=True):
        state = load_checkpoint(path)
        if (state["variant"], state["seed"], state["epoch"]) != (CORRELATION_VARIANT, seed, SCREEN_EPOCHS):
            raise ValueError(f"Unexpected expert weights: {path}")
        net = CorrelationCNN(CORRELATION_VARIANT).to(device, memory_format=torch.channels_last)
        net.load_state_dict(state["weights"])
        models.append(net)
    return models


def prediction_triplet(items, profile, table, reference, before, after, thresholds, weight, device, deadline):
    native, codec, specialist, guard = compare_views(items, profile, table, deadline)
    rows = torch.arange(len(items))
    base = reference.score(native, codec, guard, rows, deadline)
    original = recenter(scores(before, specialist, rows, device, deadline), thresholds[0])
    replacement = recenter(scores(after, specialist, rows, device, deadline), thresholds[1])
    return base, (1 - weight) * base + weight * original, (1 - weight) * base + weight * replacement


def metrics(labels, probabilities):
    return compute_metrics(labels, np.column_stack((1 - probabilities, probabilities)))


def write_csv(path, ids, scores):
    with path.open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", "label"))
        writer.writerows(zip(ids, (scores > .5).astype(int), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 60 * 60
    manifest = read_json(BUNDLE / "manifest.json")
    original_path = BUNDLE / manifest["original_submission"]
    export_summary = read_json(BUNDLE / manifest["export_summary"])
    new_summary = read_json(NEW / "summary.json")
    samples, labels, sizes, _, _, full, table, split = prepare_data()
    if split != export_summary["split_sha256"] or split != new_summary["split_sha256"]:
        raise ValueError("Checkpoints use a different labeled split")
    assets = BUNDLE / "lqy/runs"
    reference = ReleasedReference(SimpleNamespace(
        narrow_reference=assets / "codec-final/upstream-trained",
        near_reference=assets / "codec-final/near-upstream",
        night_reference=assets / "night-train/2026-11-21",
        spectrum_reference=assets / "spectrum-patch/round-01",
    ), device, split)
    official_thresholds = manifest["thresholds"]
    np.testing.assert_array_equal([item[1] for item in reference.parts],
                                  [official_thresholds["standard_jpeg"], official_thresholds["near_jpeg"]])
    old = restore_candidate(assets / "resize-specialist/round-01/full" / CORRELATION_VARIANT,
                            CORRELATION_VARIANT, CorrelationCNN, device)
    new = branch([NEW / f"s{seed}" / "last.pt" for seed in SEEDS], device)
    thresholds = official_thresholds["resize_specialist"], new_summary["expert_threshold"]
    weight = export_summary["selected_weight"]
    if weight != .25 or official_thresholds["final_label"] != .5:
        raise ValueError("Unexpected released blend or decision threshold")

    test = read_test(TEMPLATE, TEST_DIR)
    ids = [row.image_id for row in test]
    with original_path.open(newline="") as file:
        original = list(csv.DictReader(file))
    if len(ids) != 1000 or [row["image_id"] for row in original] != ids:
        raise ValueError("Saved predictions do not match the current template")
    base, old_scores, new_scores = prediction_triplet(
        test, "native", table, reference, old, new, thresholds, weight, device, deadline)
    old_labels = np.array([int(row["label"]) for row in original])
    differences = np.flatnonzero((old_scores > .5) != old_labels)
    write_csv(args.output / "exp-lqy-original-0628-replay.csv", ids, old_scores)
    np.savez(args.output / "test_scores.npz", ids=ids, base=base, original=old_scores, replacement=new_scores)
    report = {"split_sha256": split, "expert_weight": weight, "thresholds": thresholds,
              "replay_mismatches": [ids[index] for index in differences],
              "replacement_changed_predictions": int(((new_scores > .5) != old_labels).sum())}

    def record():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    record()
    print(f"Replay: {len(differences)} of 1000 labels differ.", flush=True)
    _, confirmation = validation_halves(samples, labels, sizes, len(full))
    val = [samples[index] for index in confirmation]
    truth = labels[confirmation].numpy()
    pairs, results, saved = [], {}, {}
    for profile in PROFILES:
        base, prev, replacement = prediction_triplet(
            val, profile, table, reference, old, new, thresholds, weight, device, deadline)
        pairs.append((prev, replacement))
        saved.update({f"{profile}_{part}": score for part, score in
                      (("base", base), ("original", prev), ("replacement", replacement))})
        results[profile] = {"original": metrics(truth, prev), "replacement": metrics(truth, replacement)}
        print(profile, results[profile]["original"]["accuracy"],
              results[profile]["replacement"]["accuracy"], flush=True)
    np.savez(args.output / "validation_scores.npz", ids=[row.image_id for row in val],
             labels=truth, **saved)
    gain = profile_gain(truth, pairs)
    report.update(confirmation=results, mean_gain=gain)
    report["exported_candidate"] = bool(
        not len(differences) and gain["mean_gain"] >= .01 and gain["paired_95_percent"][0] > 0
        and all(item["replacement"]["accuracy"] >= item["original"]["accuracy"] - .010000001
                for item in results.values()))
    if report["exported_candidate"]:
        write_csv(args.candidate, ids, new_scores)
    record()
    print(json.dumps(gain), flush=True)


if __name__ == "__main__":
    main()

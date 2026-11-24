"""Rescreen existing checkpoints and evaluate the winner on the fixed holdout."""

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_native_texture_base import (
    EPOCHS, calibrate_image, convert, decision_scores, export_candidate,
    matched_real_recall, reference_scores, restore, score_many, stress,
)
from lqy.codec_anchor import prepare_data
from lqy.codec_canonical import calibration_rules, validation_halves
from lqy.codec_final import profile_gain
from lqy_replay import PROFILES, metrics, read_json
from lqy_resize_retrain import training_split

ROOT = Path("fqm/runs")
RUNS = (
    "native_texture_base_20261122", "resampling_texture_base_7144",
    "resize_jpeg_weighted_texture_base_7148", "wide_native_texture_base_7151",
    "deep_native_texture_base_7153",
)
REFERENCE = ROOT / "native_texture_base_eval_20261122"
REAL_RECALL = .9175


def rank_checkpoints(models, rows, labels, table, device):
    scores = score_many(models, rows, table, partial(stress, "resize_jpeg"), device)
    real = labels == 0
    target = int(np.ceil(REAL_RECALL * real.sum()))
    results = {}
    for key, values in scores.items():
        cutoff = np.sort(values[real])[target - 1]
        results[key] = {"recall_real": float(np.mean(values[real] <= cutoff)),
                        "recall_fake": float(np.mean(values[~real] > cutoff)),
                        "auroc": metrics(labels, values)["auroc"]}
        print(key, results[key], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    samples, labels, sizes, _, _, full, table, split = prepare_data()
    selection, heldout = validation_halves(samples, labels, sizes, len(full))
    models = {}
    for name in RUNS:
        run = ROOT / name
        report = read_json(run / "summary.json")
        if report["split_sha256"] != split or report["stage"] != "training_complete":
            raise ValueError(f"Expected a completed run on the fixed split: {run}")
        for epoch in EPOCHS:
            models[name, epoch] = restore(run, epoch, device, variant=report["variant"])

    selection_rows = convert([samples[index] for index in selection])
    ranks = rank_checkpoints(models, selection_rows, labels[selection].numpy(), table, device)
    best = max(ranks, key=lambda key: (ranks[key]["recall_fake"], ranks[key]["auroc"], -key[1]))
    print("Selected", best, flush=True)
    model = {best: models[best]}
    _, anchors = training_split([samples[index] for index in full.tolist()])
    anchor_labels = np.array([row.label for row in anchors])
    calibration = score_many(model, convert(anchors), table, calibrate_image, device)[best]
    threshold = calibration_rules(calibration, anchor_labels)["both_class"]

    truth = labels[heldout].numpy()
    rows = convert([samples[index] for index in heldout])
    ids = [row["image_id"] for row in rows]
    references = reference_scores(ids, truth, REFERENCE)
    held, pairs, all_scores = {}, [], {}
    for profile in PROFILES:
        raw = score_many(model, rows, table, partial(stress, profile), device)[best]
        after = decision_scores(raw, threshold)
        before = references[profile]
        held[profile] = {"baseline": metrics(truth, before), "candidate": metrics(truth, after)}
        if profile == "alternate_resize_jpeg":
            held[profile]["matched_real_recall"] = matched_real_recall(truth, before, after)
        pairs.append((before, after))
        all_scores[f"{profile}_single"] = after
        all_scores[f"{profile}_baseline"] = before
        print(profile, held[profile], flush=True)

    report = {"baseline_online": .705, "selected_run": best[0], "selected_epoch": best[1],
              "selected_threshold": threshold, "selection_profile": "resize_jpeg",
              "selection_target_real_recall": REAL_RECALL,
              "selection": {f"{name}:{epoch}": value for (name, epoch), value in ranks.items()},
              "confirmation": held, "gain": profile_gain(truth, pairs)}
    np.savez(args.output / "validation_scores.npz", ids=ids, labels=truth, **all_scores)
    if best != (RUNS[0], 18):
        export_candidate(args.candidate, score_many, models[best], best, threshold, table, device)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

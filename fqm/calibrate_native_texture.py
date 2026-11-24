"""Calibrate the four-view single model on labeled anchors and export one candidate."""
import argparse
import csv
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_native_texture_base import convert, decision_scores, restore, score_many, stress
from lqy.codec_patch import labeled_data, tables_by_frequency
from lqy.codec_final import profile_gain
from lqy_replay import PROFILES, metrics, read_json, write_csv
from lqy_resize_retrain import TEMPLATE, TEST_DIR, identity, read_test_ids, training_split

RUN = Path("fqm/runs/native_texture_four_views_20261123")
EVAL = Path("fqm/runs/native_texture_four_views_eval_20261123")
CALIBRATION = ("native", "jpeg83", "resize_jpeg", "webp")
HARD = "resize_jpeg"


def choose_cutoff(scores, labels, old):
    points = np.unique(np.r_[1e-6, old, np.concatenate(list(scores.values())), 1 - 1e-6]).clip(1e-6, 1 - 1e-6)

    def accuracy(values):
        real = np.sort(values[labels == 0])
        fake = np.sort(values[labels == 1])
        return (np.searchsorted(real, points, side="right")
                + len(fake) - np.searchsorted(fake, points, side="right")) / len(labels)

    values = {case: accuracy(p) for case, p in scores.items()}
    baseline_index = np.searchsorted(points, old)
    usable = values[HARD] >= values[HARD][baseline_index] - .005000001
    average = np.stack(list(values.values())).mean(axis=0)
    rank = np.lexsort((np.abs(np.log(points) - np.log(old)), -values[HARD], -average))
    index = next(index for index in rank if usable[index])
    return float(points[index]), {case: {"old": float(value[baseline_index]),
                                         "new": float(value[index])} for case, value in values.items()}


def undo_recenter(values, cutoff):
    values = np.asarray(values, dtype=np.float64)
    return np.where(values > .5, cutoff + (2 * values - 1) * (1 - cutoff), 2 * values * cutoff)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    if args.candidate.exists():
        raise FileExistsError(args.candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    training = read_json(RUN / "summary.json")
    evaluated = read_json(EVAL / "summary.json")
    epoch = evaluated["selected_epoch"]
    original = evaluated["selected_threshold"]
    data, details = labeled_data()
    if training["split_sha256"] != details["split_sha256"]:
        raise ValueError("Wrong training split")
    _, anchors = training_split(data["train"])
    model = {epoch: restore(RUN, epoch, device)}
    tables, _ = tables_by_frequency(data["train"])
    table = tables[0]
    scores = {}
    for case in CALIBRATION:
        print(f"Calibration: {case}", flush=True)
        scores[case] = score_many(model, convert(anchors), table, partial(stress, case), device)[epoch]
    cutoff, calibration = choose_cutoff(scores, np.array([row.label for row in anchors]), original)

    with np.load(EVAL / "validation_scores.npz") as saved:
        truth = saved["labels"]
        confirmation, pairs = {}, []
        for case in PROFILES:
            previous = saved[f"{case}_single"]
            reference = saved[f"{case}_baseline"]
            raw = undo_recenter(previous, original)
            updated = decision_scores(raw, cutoff)
            confirmation[case] = {"old_threshold": metrics(truth, previous), "new_threshold": metrics(truth, updated),
                                  "baseline_0705": metrics(truth, reference)}
            pairs.append((reference, updated))
    gain = profile_gain(truth, pairs)
    report = {"epoch": epoch, "old_threshold": original, "threshold": cutoff,
              "calibration": calibration, "confirmation": confirmation, "gain_from_0705": gain}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"threshold": cutoff, "gain": gain}), flush=True)

    rows = read_test_ids(TEMPLATE, TEST_DIR)
    ids = [row["image_id"] for row in rows]
    with TEMPLATE.open(newline="") as file:
        template_ids = [row["image_id"] for row in csv.DictReader(file)]
    assert len(ids) == len(set(ids)) == 1000 and ids == template_ids
    raw = score_many(model, rows, table, identity, device)[epoch]
    if not np.isfinite(raw).all():
        raise ValueError("Invalid test probabilities")
    adjusted = decision_scores(raw, cutoff)
    write_csv(args.candidate, ids, adjusted)
    with args.candidate.open(newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["image_id", "label"]
        exported = list(reader)
    assert [row["image_id"] for row in exported] == ids
    np.testing.assert_array_equal([int(row["label"]) for row in exported], adjusted > .5)
    np.savez(args.output / "test_scores.npz", ids=ids, probabilities=raw, threshold=cutoff)
    print(f"Verified CSV: {args.candidate}", flush=True)


if __name__ == "__main__":
    main()

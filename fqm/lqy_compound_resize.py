"""Train a resize-plus-recompression specialist against the verified 0.628 baseline."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from detection.data import jpeg_roundtrip
from lqy.canonical_augmentation import CORRELATION_VARIANT, CorrelationCNN, canonical_views
from lqy.codec_anchor import prepare_data, recenter
from lqy.codec_canonical import calibration_rules, transform, validation_halves
from lqy.codec_final import profile_gain
from lqy.night_train import SCREEN_EPOCHS, SEEDS, fit
from lqy_replay import PROFILES, metrics, read_json, write_csv
from lqy_resize_retrain import (TEMPLATE, TEST_DIR, identity, read_test_ids,
                                specialist_scores, training_split)
from lqy_specialist_gate import REPLAY

KERNELS = (Image.Resampling.BICUBIC, Image.Resampling.LANCZOS, Image.Resampling.BILINEAR)


def acquisition(image, assignment):
    position = assignment[0] / 255
    if assignment[1] % 7:
        scale = .68 + .26 * position if min(image.size) >= 160 else 1.10 + .30 * position
        size = tuple(max(1, round(axis * scale)) for axis in image.size)
        image = image.resize(size, KERNELS[assignment[1] % len(KERNELS)])
    return jpeg_roundtrip(image, quality=69 + assignment[2] % 23, subsampling=assignment[3] % 3)


def mixed_calibration(image, image_id):
    assignment = sha256(f"compound-calibration-v1:{image_id}".encode()).digest()
    return acquisition(image, assignment) if assignment[4] % 2 else image


def package(samples):
    return [{"image_id": row.image_id, "path": row.path} for row in samples]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 55 * 60
    samples, labels, sizes, _, _, full, table, split = prepare_data()
    assert split == read_json(REPLAY / "summary.json")["split_sha256"]
    train, anchors = training_split([samples[i] for i in full.tolist()])
    first, second = canonical_views(train, table, deadline, paired=True, alter_both=True,
                                    acquisition=acquisition)
    y = torch.tensor([row.label for row in train])
    models = [fit(args.output / f"s{seed}", CORRELATION_VARIANT, seed, SCREEN_EPOCHS, first, y,
                  torch.arange(len(train)), device, deadline, model_factory=CorrelationCNN, paired_data=second)
              for seed in SEEDS]
    del first, second
    calibration = specialist_scores(models, package(anchors), table, mixed_calibration, device, deadline)
    threshold = calibration_rules(calibration, np.array([row.label for row in anchors]))["both_class"]
    _, held = validation_halves(samples, labels, sizes, len(full))
    val = [samples[i] for i in held]
    truth = labels[held].numpy()
    with np.load(REPLAY / "validation_scores.npz") as cache:
        np.testing.assert_array_equal(cache["ids"], [row.image_id for row in val])
        np.testing.assert_array_equal(cache["labels"], truth)
        original = {case: (cache[f"{case}_base"], cache[f"{case}_original"]) for case in PROFILES}
    results, pairs, scored = {}, [], {}
    for case in PROFILES:
        def perturb(image, _):
            return transform(image, case)
        print(f"Validating {case}", flush=True)
        values = recenter(specialist_scores(models, package(val), table, perturb, device, deadline), threshold)
        base, reference = original[case]
        candidate = .75 * base + .25 * values
        results[case] = {"original": metrics(truth, reference), "candidate": metrics(truth, candidate)}
        pairs.append((reference, candidate))
        scored[case] = candidate
    gain = profile_gain(truth, pairs)
    delta = {case: row["candidate"]["accuracy"] - row["original"]["accuracy"]
             for case, row in results.items()}
    accepted = (gain["mean_gain"] >= .01 and gain["paired_95_percent"][0] > 0
                and delta["alternate_resize_jpeg"] >= .02 and min(delta.values()) >= -.010000001)
    report = {"source_commit": "b23beec", "split_sha256": split, "threshold": threshold, "gain": gain,
              "accuracy_change": delta, "validation": results, "candidate_exported": False,
              "train_count": len(train), "calibration_count": len(anchors), "seeds": SEEDS,
              "epochs": SCREEN_EPOCHS}
    np.savez(args.output / "validation_scores.npz", ids=[row.image_id for row in val], labels=truth, **scored)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gain": gain, "accuracy_change": delta}, indent=2), flush=True)
    if accepted:
        rows = read_test_ids(TEMPLATE, TEST_DIR)
        ids = [row["image_id"] for row in rows]
        with np.load(REPLAY / "test_scores.npz") as cache, TEMPLATE.open(newline="") as file:
            expected = [row["image_id"] for row in csv.DictReader(file)]
            assert ids == expected and len(ids) == len(set(ids)) == 1000
            np.testing.assert_array_equal(cache["ids"], ids)
            base, reference = cache["base"], cache["original"]
        expert = specialist_scores(models, rows, table, identity, device, deadline)
        candidate = .75 * base + .25 * recenter(expert, threshold)
        changed = int(((candidate > .5) != (reference > .5)).sum())
        np.savez(args.output / "test_scores.npz", ids=ids, baseline=reference, candidate=candidate)
        if changed:
            write_csv(args.candidate, ids, candidate)
            report.update(candidate_exported=True, changed_test_predictions=changed)
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"Candidate ready: {args.candidate}", flush=True)


if __name__ == "__main__":
    main()

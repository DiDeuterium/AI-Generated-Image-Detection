"""Export the verified 0.644 base augmented by the compound-resize specialist."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from lqy.codec_anchor import recenter
from lqy.codec_final import profile_gain
from lqy.codec_patch import labeled_data, tables_by_frequency
from lqy.night_train import SEEDS
from lqy_replay import PROFILES, branch, metrics, read_json, write_csv
from lqy_resize_retrain import TEMPLATE, TEST_DIR, identity, read_test_ids, specialist_scores

REPLAY = Path("fqm/runs/lqy_replay_20261122")
COMPOUND = Path("fqm/runs/lqy_compound_resize_20261122")
REFERENCE_CSV = Path("fqm/submissions/exp-lqy-expert45-20261122.csv")


def blend(base, resize25, compound25):
    resize = (resize25 - .75 * base) / .25
    compound = (compound25 - .75 * base) / .25
    return .55 * base + .45 * resize, .35 * base + .45 * resize + .20 * compound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.candidate.exists():
        raise FileExistsError("Use new output paths")
    torch.set_num_threads(4)
    summary = read_json(COMPOUND / "summary.json")
    with np.load(REPLAY / "validation_scores.npz") as ref, np.load(COMPOUND / "validation_scores.npz") as compound:
        for key in ("ids", "labels"):
            np.testing.assert_array_equal(ref[key], compound[key])
        truth = ref["labels"]
        pairs = [blend(ref[f"{case}_base"], ref[f"{case}_replacement"], compound[case]) for case in PROFILES]
    report = {"baseline_online": .644, "weights": {"base": .35, "resize_expert": .45, "compound_expert": .20},
              "profiles": {case: {"baseline": metrics(truth, before), "combined": metrics(truth, after)}
                           for case, (before, after) in zip(PROFILES, pairs, strict=True)},
              "gain": profile_gain(truth, pairs),
              "validation_note": "Exploratory re-use of validation images for combination selection."}

    split, details = labeled_data()
    if details["split_sha256"] != summary["split_sha256"]:
        raise ValueError("Trained model split does not match current data")
    tables, _ = tables_by_frequency(split["train"])
    device = torch.device("cuda")
    models = branch([COMPOUND / f"s{seed}" / "last.pt" for seed in SEEDS], device)
    rows = read_test_ids(TEMPLATE, TEST_DIR)
    ids = [row["image_id"] for row in rows]
    with REFERENCE_CSV.open(newline="") as file:
        archived = list(csv.DictReader(file))
    with np.load(REPLAY / "test_scores.npz") as data:
        np.testing.assert_array_equal(data["ids"], ids)
        base = data["base"]
        resize25 = data["replacement"]
    if len(ids) != 1000 or len(set(ids)) != 1000 or ids != [row["image_id"] for row in archived]:
        raise ValueError("Reference submission order differs from current template")
    expert = recenter(specialist_scores(models, rows, tables[0], identity, device,
                                        time.monotonic() + 8 * 60), summary["threshold"])
    reference, candidate = blend(base, resize25, .75 * base + .25 * expert)
    np.testing.assert_array_equal(reference > .5, [int(row["label"]) for row in archived])
    if not np.isfinite(candidate).all():
        raise ValueError("Invalid candidate score")
    changes = int(((reference > .5) != (candidate > .5)).sum())
    args.output.mkdir()
    np.savez(args.output / "test_scores.npz", ids=ids, baseline=reference, candidate=candidate)
    report["changed_test_predictions"] = changes
    if changes:
        write_csv(args.candidate, ids, candidate)
        with args.candidate.open(newline="") as file:
            exported = list(csv.DictReader(file))
        np.testing.assert_array_equal([row["image_id"] for row in exported], ids)
        np.testing.assert_array_equal([int(row["label"]) for row in exported], candidate > .5)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gain": report["gain"], "changes": changes}), flush=True)


if __name__ == "__main__":
    main()

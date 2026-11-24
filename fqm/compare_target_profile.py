"""Summarize paired held-out scores for the equal-budget alignment experiment."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_target_profile import CASES, RUNS, SMOOTHED_CASE

GROUPS = ("width500", "other")


def read_case(directory, name):
    # The audit writes flat outputs for multiple models and nested outputs when
    # invoked once per run.
    stem = directory / name
    if not stem.with_suffix(".json").is_file():
        stem = stem / name
    report = json.loads(stem.with_suffix(".json").read_text())
    run = json.loads((RUNS / name / "results.json").read_text())
    with np.load(stem.with_suffix(".npz")) as source:
        scores = {key: source[key] for key in source.files}
    return report, run, scores


def paired_deltas(control, matched):
    for key in ("ids", "labels", "cohorts"):
        if not np.array_equal(control[key], matched[key]):
            raise ValueError(f"Paired holdout rows differ in {key}")
    ids, labels, cohorts = [control[key] for key in ("ids", "labels", "cohorts")]
    if len(set(ids)) != len(ids) or len(ids) != 800:
        raise ValueError("Expected 800 unique held-out rows")
    rng = np.random.default_rng(20261121)
    report = {}
    for case in (*CASES, SMOOTHED_CASE):
        if case not in control and case not in matched:
            continue
        difference = (((matched[case] > .5) == labels).astype(int)
                      - (((control[case] > .5) == labels).astype(int)))
        groups = {}
        for name in ("all", *GROUPS):
            included = np.ones(len(labels), dtype=bool) if name == "all" else cohorts == name
            strata = [np.flatnonzero(included & (cohorts == group) & (labels == label))
                      for group in GROUPS for label in (0, 1)]
            samples = np.concatenate([rng.choice(stratum, (2000, len(stratum)), replace=True)
                                      for stratum in strata if stratum.size], axis=1)
            groups[name] = {"delta": float(difference[included].mean()),
                            "ci95": np.quantile(difference[samples].mean(axis=1), [.025, .975]).tolist()}
        report[case] = groups
    return report


def processing_flips(scores):
    labels = scores["labels"]
    report = {
        case: flip_counts(labels, scores["orig"], scores[case])
        for case in (*CASES[1:], SMOOTHED_CASE) if case in scores
    }
    if SMOOTHED_CASE in scores and "both" in scores:
        report["smoothing_only"] = flip_counts(labels, scores["both"], scores[SMOOTHED_CASE])
    return report


def flip_counts(labels, before, after):
    original = before > .5
    changed = original != (after > .5)
    report = {}
    for name, mask in (("all", np.ones(len(labels), dtype=bool)),
                       ("real", labels == 0), ("fake", labels == 1)):
        flips = changed & mask
        report[name] = {
            "count": int(flips.sum()),
            "rate": float(flips.sum() / mask.sum()),
            "correct_to_wrong": int((flips & (original == labels)).sum()),
            "wrong_to_correct": int((flips & (original != labels)).sum()),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("control")
    parser.add_argument("matched")
    args = parser.parse_args()
    baseline, trial = (read_case(args.directory, name) for name in (args.control, args.matched))
    if baseline[1]["total_updates"] != trial[1]["total_updates"]:
        raise ValueError("Training budgets differ")
    gpu = baseline[1]["environment"]["gpu"]
    if trial[1]["environment"]["gpu"] != gpu:
        raise ValueError("Training GPU types differ")
    report = {"metrics": {"control": baseline[0], "matched": trial[0]}, "gpu": gpu,
              "model_selection_epochs": [run[1]["best_epoch"] for run in (baseline, trial)],
              "paired_deltas": paired_deltas(baseline[2], trial[2]),
              "processing_flips": {"control": processing_flips(baseline[2]),
                                   "matched": processing_flips(trial[2])}}
    output = args.directory / "comparison.json"
    with output.open("x") as file:
        json.dump(report, file, indent=2)
    print(report["paired_deltas"]["both"], flush=True)


if __name__ == "__main__":
    main()

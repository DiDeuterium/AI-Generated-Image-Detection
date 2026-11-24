"""Select a frozen two-model inference policy with a class-balanced audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_proxy_shift import ROOT
from compare_target_profile import paired_deltas
from ssd.data import read_splits
from ssd.target_profile import validation_groups


CASES = ("orig", "geometry", "encoding", "both", "legacy_strong")
MODELS = ("target_control_20261121", "target_matched_20261121")


def read_scores(directory):
    results = []
    for name in MODELS:
        with np.load(directory / f"{name}.npz") as data:
            results.append({key: data[key] for key in data.files})
    left, right = results
    for key in ("ids", "labels", "cohorts"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"Mismatched {key}")
    average = {**left, **{case: (left[case] + right[case]) * .5 for case in CASES}}
    return average, right


def score(scores, keep):
    metrics = {}
    labels = scores["labels"]
    cohorts = scores["cohorts"]
    for case in CASES:
        pred = scores[case] > .5
        metrics[case] = {
            "all": float(np.mean(labels[keep] == pred[keep])),
            "width500": float(np.mean(labels[keep & (cohorts == "width500")]
                                      == pred[keep & (cohorts == "width500")])),
            "other": float(np.mean(labels[keep & (cohorts == "other")]
                                   == pred[keep & (cohorts == "other")])),
            "recall_real": float(np.mean(~pred[keep & (labels == 0)])),
            "recall_fake": float(np.mean(pred[keep & (labels == 1)])),
        }
    return metrics


def robustness(metrics):
    worst = min(metrics["both"]["width500"], metrics["both"]["other"],
                metrics["legacy_strong"]["all"])
    return worst, metrics["both"]["all"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    center, baseline = read_scores(args.root / "center")
    five, _ = read_scores(args.root / "five")
    for field in ("ids", "labels", "cohorts"):
        if not np.array_equal(center[field], five[field]):
            raise ValueError(f"Inference methods did not use the same {field}")
    _, rows = read_splits(ROOT / "splits/train_val.csv", ROOT / "dataset")
    _, held = validation_groups(rows)
    held = np.isin(center["ids"], [row["image_id"] for row in held])
    assert int(held.sum()) == 800 and len(held) == 1600

    variants = {"center": center, "five": five}
    fit = {name: score(scores, ~held) for name, scores in variants.items()}
    results = {name: score(scores, held) for name, scores in variants.items()}
    results["baseline"] = score(baseline, held)
    winner = max(fit, key=lambda name: robustness(fit[name]))
    baseline_robustness = robustness(results["baseline"])
    center_worst = robustness(results["center"])[0]
    available = [name for name in (winner, "center")
                 if robustness(results[name])[0] > baseline_robustness[0]
                 and results[name]["both"]["all"] >= baseline_robustness[1] - .02
                 and (name == "center" or robustness(results[name])[0] > center_worst)]
    selected = available[0] if available else None
    prior = {key: values[held] for key, values in baseline.items()}
    report = {"selection": fit, "heldout": results, "selection_winner": winner,
              "export": selected, "versus_baseline": {
                  name: paired_deltas(prior, {key: values[held] for key, values in scores.items()})
                  for name, scores in variants.items()
              }}
    with (args.root / "comparison.json").open("x") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(selected or "none", flush=True)


if __name__ == "__main__":
    main()

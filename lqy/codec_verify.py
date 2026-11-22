"""Check the frozen JPEG-response rule on the untouched 256px labeled cohort."""

import argparse
from pathlib import Path
import time

import torch

from detection.data import read_test
from detection.training import write_csv, write_json

from .cnn import device_for, read_json, run_lock
from .codec_anchor import cutoff, exclude_anchor, prepare_data, recenter, view_pair
from .codec_response import VARIANT, train_variant
from .codec_scale import baseline
from .deep_fusion import comparison
from .night_train import scores


COHORT = 256
RULE = "cross_fit"
MIN_INDEPENDENT_GAIN = 0.05


def run(args, device, deadline):
    data = prepare_data((512, 1024, 128, COHORT))
    samples, labels, sizes, anchor, folds, full, table, identity = data
    previous = read_json(args.anchor_reference / "summary.json")
    if previous["split_sha256"] != identity or not previous["gates"][RULE]["passes"]:
        raise ValueError("The frozen rule must pass its initial labeled screening")
    quantile = previous["quantiles"][RULE]
    prior = {group: results[RULE] for group, results in previous["screen"].items()}
    prior["128"] = previous["confirmation"][RULE]
    report = {"split_sha256": identity, "quantile": quantile, "earlier_results": prior,
              "minimum_independent_gain": MIN_INDEPENDENT_GAIN}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("confirm_256")
    train, val = folds[COHORT]
    train = exclude_anchor(train, labels, anchor)
    rows = torch.cat((train, anchor, val))
    native, paired = view_pair([samples[i] for i in rows.tolist()], table, deadline)
    end_anchor = len(train) + len(anchor)
    anchor_rows = torch.arange(len(train), end_anchor)
    validation = torch.arange(end_anchor, len(rows))
    models = train_variant(args.output / "confirm_256" / VARIANT, VARIANT, paired, labels[rows], sizes,
                           torch.arange(len(train)), device, deadline)
    threshold = cutoff(scores(models, paired, anchor_rows, device, deadline), quantile)
    response = scores(models, paired, validation, device, deadline)
    reference = baseline(args, COHORT, native[validation], labels[val], device, deadline)
    result = comparison(labels[val].numpy(), reference, recenter(response, threshold))
    report["confirmation_256"] = {"threshold": threshold, "train_count": len(train), **result}
    print(f"256: accuracy {result['reference']['accuracy']:.3%} -> {result['deep_fusion']['accuracy']:.3%}, "
          f"AUROC {result['reference']['auroc']:.4f} -> {result['deep_fusion']['auroc']:.4f}", flush=True)
    del native, paired, models

    gain = result["delta"]["accuracy"]
    previous_gains = [item["delta"]["accuracy"] for item in prior.values()]
    report["mean_four_cohort_gain"] = (sum(previous_gains) + gain) / (len(previous_gains) + 1)
    reliable = (min(previous_gains) >= 0 and report["mean_four_cohort_gain"] >= MIN_INDEPENDENT_GAIN
                and gain >= MIN_INDEPENDENT_GAIN and result["delta"]["auroc"] > 0
                and result["paired_accuracy"]["bootstrap_95_percent"][0] > 0)
    if not reliable:
        record("stopped_at_confirmation")
        return

    record("full_training")
    train = exclude_anchor(full, labels, anchor)
    rows = torch.cat((train, anchor))
    native, paired = view_pair([samples[i] for i in rows.tolist()], table, deadline)
    models = train_variant(args.output / "full" / VARIANT, VARIANT, paired, labels[rows], sizes,
                           torch.arange(len(train)), device, deadline)
    threshold = cutoff(scores(models, paired, torch.arange(len(train), len(rows)), device, deadline), quantile)
    report["train_count"] = len(train)
    report["full_threshold"] = threshold
    del native, paired

    record("export")
    tests = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
    if len(tests) != 1000:
        raise ValueError("Expected 1,000 test images")
    _, paired = view_pair(tests, table, deadline)
    values = scores(models, paired, torch.arange(len(tests)), device, deadline)
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(value > threshold)) for sample, value in zip(tests, values, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--anchor-reference", type=Path, default=Path("lqy/runs/codec-anchor/round-01"))
    parser.add_argument("--group-reference", type=Path, default=Path("lqy/runs/group-patch/round-01"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

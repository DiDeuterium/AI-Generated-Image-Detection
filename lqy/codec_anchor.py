"""Calibrate the JPEG-response CNN against photographs it did not train on."""

import argparse
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from detection.data import read_test
from detection.training import load_checkpoint, write_csv, write_json

from .cnn import device_for, read_json, run_lock
from .codec_patch import cache_encodes, labeled_data, tables_by_frequency
from .codec_response import CodecResponseCNN, PairedViews, VARIANT, train_variant
from .deep_fusion import CONFIRM, MIN_CONFIRM_GAIN, SCREEN, combined_reference, comparison
from .night_train import BATCH, CROP, SEEDS, cache, sample_rows, scores
from .offline_search import gate
from .spectrum_patch import QUALITY


ANCHOR_SIZE = 320
FIXED_QUANTILE = 0.80
QUANTILES = np.linspace(0.60, 0.95, 36)


def prepare_data(cohorts=(*SCREEN, CONFIRM)):
    split, details = labeled_data()
    samples = split["train"] + split["val"]
    end = len(split["train"])
    labels = torch.tensor([sample.label for sample in samples])
    sizes = []
    for sample in samples:
        with Image.open(sample.path) as image:
            sizes.append(image.size)
    real_train = sorted((i for i in range(end) if samples[i].label == 0),
                       key=lambda i: sha256(f"cohort-probe:{samples[i].image_id}".encode()).digest())
    anchor = torch.tensor(real_train[-ANCHOR_SIZE:])
    real_val = sorted((i for i in range(end, len(samples)) if samples[i].label == 0),
                     key=lambda i: sha256(f"threshold-audit:{samples[i].image_id}".encode()).digest())
    splits = {}
    default_train, _ = sample_rows(samples, sizes, end)
    for cohort in cohorts:
        train, group = sample_rows(samples, sizes, end, cohort)
        fake = group[len(group) // 2:].tolist()
        real = real_val[:len(fake)]
        if len(real) != len(fake):
            raise ValueError("Insufficient disjoint labeled validation images")
        real_val = real_val[len(fake):]
        splits[cohort] = train, torch.tensor(real + fake)
    table = tables_by_frequency(split["train"])[0][0]
    return samples, labels, sizes, anchor, splits, default_train, table, details["split_sha256"]


def view_pair(items, table, deadline, *, crop_size=CROP):
    native, _ = cache(items, QUALITY, deadline, crop_size=crop_size)
    alternate = cache_encodes(items, [table] * len(items), deadline, crop_size=crop_size)
    return native, PairedViews(native, alternate)


def cutoff(anchor_scores, quantile):
    return float(np.clip(np.quantile(anchor_scores, quantile), 1e-6, 1 - 1e-6))


def recenter(values, threshold):
    return np.where(values > threshold, 0.5 + 0.5 * (values - threshold) / (1 - threshold),
                    0.5 * values / threshold)


def choose_quantile(folds):
    def rank(q):
        accuracies = [np.mean((score > cutoff(anchor_scores, q)) == truth)
                      for truth, _, score, anchor_scores in folds]
        return min(accuracies), np.mean(accuracies), -abs(q - FIXED_QUANTILE)

    return float(max(QUANTILES, key=rank))


def restore(args, cohort, device):
    result = []
    for seed in SEEDS:
        path = args.response / f"group-{cohort}" / VARIANT / f"s{seed}" / "last.pt"
        state = load_checkpoint(path)
        if (state["variant"], state["seed"], state["epoch"]) != (VARIANT, seed, 6):
            raise ValueError(f"Unexpected JPEG-response checkpoint: {path}")
        candidate = CodecResponseCNN(VARIANT).to(device, memory_format=torch.channels_last)
        candidate.load_state_dict(state["weights"])
        result.append(candidate)
    return result


def screen(args, device, deadline, data):
    samples, labels, _, anchor, splits, _, table, identity = data
    if read_json(args.response / "summary.json")["split_sha256"] != identity:
        raise ValueError("JPEG-response checkpoints use a different labeled split")
    _, anchor_data = view_pair([samples[i] for i in anchor.tolist()], table, deadline)
    anchor_positions = torch.arange(len(anchor))
    groups = {}
    for cohort in SCREEN:
        train, val = splits[cohort]
        if torch.isin(anchor, train).any():
            raise ValueError(f"Photograph anchor overlaps the training set of cohort {cohort}")
        native, paired = view_pair([samples[i] for i in val.tolist()], table, deadline)
        rows = torch.arange(len(val))
        baseline = combined_reference(args, cohort, native, labels[val], None, rows, device, deadline)
        model = restore(args, cohort, device)
        groups[cohort] = (labels[val].numpy(), baseline, scores(model, paired, rows, device, deadline),
                          scores(model, anchor_data, anchor_positions, device, deadline))

    report = {"split_sha256": identity, "anchor_count": len(anchor), "screen": {},
              "quantiles": {"fixed": FIXED_QUANTILE, "cross_fit": choose_quantile(list(groups.values()))}}
    paired = {name: [] for name in report["quantiles"]}
    for target, source in (SCREEN, SCREEN[::-1]):
        truth, baseline, predicted, anchor_scores = groups[target]
        results = {}
        for name, q in (("fixed", FIXED_QUANTILE), ("cross_fit", choose_quantile([groups[source]]))):
            threshold = cutoff(anchor_scores, q)
            calibrated = recenter(predicted, threshold)
            result = comparison(truth, baseline, calibrated)
            results[name] = {"quantile": q, "threshold": threshold, **result}
            paired[name].append((truth, baseline, calibrated))
            print(f"{target} {name}: accuracy {result['reference']['accuracy']:.3%} -> "
                  f"{result['deep_fusion']['accuracy']:.3%}, quantile {q:.3f} threshold {threshold:.4f}", flush=True)
        report["screen"][str(target)] = results
    report["gates"] = {name: gate({cohort: rows[name] for cohort, rows in report["screen"].items()}, values)
                       for name, values in paired.items()}
    report["stage"] = "screen_passed" if any(item["passes"] for item in report["gates"].values()) else "stopped_at_screen"
    write_json(args.output / "summary.json", report)


def exclude_anchor(train, labels, anchor):
    eligible = train[~torch.isin(train, anchor)]
    real, fake = eligible[labels[eligible] == 0], eligible[labels[eligible] == 1]
    count = min(len(real), len(fake)) // BATCH * BATCH
    if not count:
        raise ValueError("No balanced non-anchor training batch")
    return torch.cat((real[:count], fake[:count]))


def complete(args, device, deadline, data):
    samples, labels, sizes, anchor, splits, full, table, identity = data
    report = read_json(args.output / "summary.json")
    if report["stage"] != "screen_passed" or report["split_sha256"] != identity:
        raise ValueError("Finishing requires a positive screen on the current labeled split")

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("confirm_128")
    reference_train, val = splits[CONFIRM]
    train = exclude_anchor(reference_train, labels, anchor)
    used = torch.unique(torch.cat((reference_train, train, val, anchor)), sorted=True)
    native, paired = view_pair([samples[i] for i in used.tolist()], table, deadline)
    position = lambda rows: torch.searchsorted(used, rows)
    models = train_variant(args.output / "confirm_128" / VARIANT, VARIANT, paired, labels[used], sizes,
                           position(train), device, deadline)
    reference = combined_reference(args, CONFIRM, native, labels[used], position(reference_train), position(val),
                                   device, deadline)
    anchor_scores = scores(models, paired, position(anchor), device, deadline)
    response = scores(models, paired, position(val), device, deadline)
    report["confirmation"] = {}
    selected = []
    for name, item in report["gates"].items():
        if not item["passes"]:
            continue
        threshold = cutoff(anchor_scores, report["quantiles"][name])
        result = comparison(labels[val].numpy(), reference, recenter(response, threshold))
        report["confirmation"][name] = {"threshold": threshold, **result}
        print(f"128 {name}: accuracy {result['reference']['accuracy']:.3%} -> "
              f"{result['deep_fusion']['accuracy']:.3%}", flush=True)
        if result["delta"]["accuracy"] >= MIN_CONFIRM_GAIN and result["delta"]["auroc"] > 0:
            selected.append(name)
    del native, paired, models
    if not selected:
        record("stopped_at_confirmation")
        return

    choice = max(selected, key=lambda name: report["gates"][name]["mean_gain"])
    report["selected"] = choice
    record("full_training")
    train = exclude_anchor(full, labels, anchor)
    rows = torch.cat((train, anchor))
    native, paired = view_pair([samples[i] for i in rows.tolist()], table, deadline)
    models = train_variant(args.output / "full" / VARIANT, VARIANT, paired, labels[rows], sizes,
                           torch.arange(len(train)), device, deadline)
    report["train_count"] = len(train)
    anchor_scores = scores(models, paired, torch.arange(len(train), len(rows)), device, deadline)
    threshold = cutoff(anchor_scores, report["quantiles"][choice])
    report["full_threshold"] = threshold
    del native, paired
    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    _, paired = view_pair(test, table, deadline)
    probability = scores(models, paired, torch.arange(len(test)), device, deadline)
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(value > threshold)) for sample, value in zip(test, probability, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--mode", choices=("screen", "complete"), required=True)
    parser.add_argument("--response", type=Path, default=Path("lqy/runs/codec-response/round-01"))
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        stage = screen if args.mode == "screen" else complete
        stage(args, device_for(args.device, 6), time.monotonic() + 3600, prepare_data())


if __name__ == "__main__":
    main()

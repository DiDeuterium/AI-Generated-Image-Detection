"""Train the supported hybrid detector and verify it on fixed labeled images."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from detection.data import jpeg_roundtrip, read_test
from detection.training import write_csv, write_json

from .cnn import device_for, read_json, run_lock
from .codec_anchor import cutoff, exclude_anchor, prepare_data, recenter, view_pair
from .codec_patch import encode
from .codec_response import PairedViews, train_variant
from .codec_scale import HYBRID, QUANTILE, HybridResponseCNN
from .deep_fusion import comparison
from .night_train import CROP, check_time, image_patches, scores
from .phase_patch import full_reference
from .spectrum_patch import QUALITY


UPSTREAM = (85, 70)
TRAIN_UPSTREAM = (65, 80)
WIDE_TRAIN_UPSTREAM = (62, 68, 70, 70, 72, 80, 88)
NEAR_TRAIN_UPSTREAM = (71, 72, 73, 74, 75, 76)


def stress_pair(sample, quality, table):
    with Image.open(sample.path) as image:
        rgb = jpeg_roundtrip(image.convert("RGB"), quality=quality, subsampling=2)
    first = image_patches(rgb, QUALITY)
    second = image_patches(rgb, QUALITY, encoder=partial(encode, table=table))
    return first, second


def cache_stress(items, quality, table, deadline):
    print(f"Preparing upstream JPEG Q{quality} for {len(items)} labeled images", flush=True)
    first = np.empty((len(items), 5, 3, CROP, CROP), dtype=np.uint8)
    second = np.empty_like(first)
    render = partial(stress_pair, quality=quality, table=table)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(items), 32):
            check_time(deadline)
            for i, (a, b) in enumerate(workers.map(render, items[start:start + 32]), start=start):
                first[i], second[i] = a, b
    native = torch.from_numpy(first)
    return native, PairedViews(native, torch.from_numpy(second))


def training_stress(items, table, deadline, qualities=TRAIN_UPSTREAM):
    assignments = [qualities[sha256(f"upstream-stress:{item.image_id}".encode()).digest()[0] % len(qualities)]
                   for item in items]
    print(f"Preparing training-only JPEG profiles {qualities} for {len(items)} images", flush=True)
    first = np.empty((len(items), 5, 3, CROP, CROP), dtype=np.uint8)
    second = np.empty_like(first)
    render = partial(stress_pair, table=table)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(items), 32):
            check_time(deadline)
            pairs = workers.map(render, items[start:start + 32], assignments[start:start + 32])
            for i, (a, b) in enumerate(pairs, start=start):
                first[i], second[i] = a, b
    return PairedViews(torch.from_numpy(first), torch.from_numpy(second))


def profile_gain(truth, pairs):
    changes = np.stack([((after > 0.5) == truth).astype(float) - ((before > 0.5) == truth)
                        for before, after in pairs]).mean(axis=0)
    rng = np.random.default_rng(17)
    bootstrap = np.zeros(4000)
    for value in (0, 1):
        subset = changes[truth == value]
        sampled = rng.integers(len(subset), size=(len(bootstrap), len(subset)))
        bootstrap += subset[sampled].sum(axis=1) / len(changes)
    return {"mean_gain": float(changes.mean()), "paired_95_percent": np.quantile(bootstrap, (0.025, 0.975)).tolist()}


def run(args, device, deadline):
    support = read_json(args.support_reference / "benchmark.json")
    matched = support["reference_calibrated"]
    if matched["delta"]["accuracy"] < 0.05 or matched["paired_accuracy"]["bootstrap_95_percent"][0] <= 0:
        raise ValueError("Equal-support comparison must show reliable improvement")
    samples, labels, sizes, anchor, _, full, table, identity = prepare_data()
    split_end = len(full)
    if not torch.equal(torch.sort(full).values, torch.arange(split_end)):
        raise ValueError("Unexpected official training partition")
    train = exclude_anchor(full, labels, anchor)
    packed_rows = torch.cat((train, anchor))
    val = torch.arange(split_end, len(samples))
    items = [samples[i] for i in val.tolist()]
    training_profiles = ()
    if args.upstream_training:
        training_profiles = TRAIN_UPSTREAM
    elif args.wide_upstream:
        training_profiles = WIDE_TRAIN_UPSTREAM
    elif args.near_upstream:
        training_profiles = NEAR_TRAIN_UPSTREAM
    profiles = (*UPSTREAM, 92, 67, 73) if args.wide_upstream else (*UPSTREAM, 92) if training_profiles else UPSTREAM
    report = {"split_sha256": identity, "train_count": len(train), "anchor_count": len(anchor),
              "quantile": QUANTILE, "validation_profiles": ["native", *profiles], "labeled_validation": {},
              "training_upstream_profiles": training_profiles,
              "release_rule": {"min_accuracy_each": 0.75, "max_loss_any_profile": 0.015,
                               "min_mean_accuracy_gain": 0.03, "paired_interval_lower_above_zero": True}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("training")
    train_and_anchor = [samples[i] for i in packed_rows.tolist()]
    native, paired = view_pair(train_and_anchor, table, deadline)
    alternate = training_stress(train_and_anchor, table, deadline, training_profiles) if training_profiles else None
    models = train_variant(args.output / "full" / HYBRID, HYBRID, paired, labels[packed_rows], sizes,
                           torch.arange(len(train)), device, deadline, model_factory=HybridResponseCNN, paired_data=alternate)
    anchor_rows = torch.arange(len(train), len(packed_rows))
    anchor_scores = scores(models, paired, anchor_rows, device, deadline)
    if alternate is not None:
        compressed_scores = scores(models, alternate, anchor_rows, device, deadline)
        anchor_scores = compressed_scores if args.near_upstream else np.concatenate((anchor_scores, compressed_scores))
    threshold = cutoff(anchor_scores, QUANTILE)
    report["threshold"] = threshold
    del native, paired, alternate
    if args.near_upstream:
        record("ready_for_route")
        return

    rows = torch.arange(len(val))
    truth = labels[val].numpy()
    pairs = []
    for quality in (None, *profiles):
        name = "native" if quality is None else f"upstream_q{quality}"
        record(f"validate_{name}")
        if quality is None:
            native, paired = view_pair(items, table, deadline)
        else:
            native, paired = cache_stress(items, quality, table, deadline)
        reference = full_reference(args, native, rows, device, deadline)
        candidate = recenter(scores(models, paired, rows, device, deadline), threshold)
        result = comparison(truth, reference, candidate)
        report["labeled_validation"][name] = result
        pairs.append((reference, candidate))
        print(f"{name}: accuracy {result['reference']['accuracy']:.3%} -> {result['deep_fusion']['accuracy']:.3%}",
              flush=True)
        del native, paired
    aggregate = profile_gain(truth, pairs)
    report["aggregate"] = aggregate
    accepted = (aggregate["mean_gain"] >= 0.03 and aggregate["paired_95_percent"][0] > 0
                and all(row["deep_fusion"]["accuracy"] >= 0.75 and row["delta"]["accuracy"] >= -0.015
                        for row in report["labeled_validation"].values()))
    if not accepted:
        record("stopped_at_validation")
        return

    record("export")
    tests = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
    if len(tests) != 1000:
        raise ValueError("Expected 1,000 test images")
    _, prediction_data = view_pair(tests, table, deadline)
    values = scores(models, prediction_data, torch.arange(len(tests)), device, deadline)
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(value > threshold)) for sample, value in zip(tests, values, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument("--upstream-training", action="store_true", help="Enforce agreement on two upstream JPEG qualities")
    augmentation.add_argument("--wide-upstream", action="store_true", help="Train with a wider set of upstream JPEG qualities")
    augmentation.add_argument("--near-upstream", action="store_true", help="Train a checkpoint for the near-Q75 route without test inference")
    parser.add_argument("--support-reference", type=Path, default=Path("lqy/runs/codec-support/round-01"))
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

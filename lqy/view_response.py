"""Learn from paired changes in scores under controlled image resampling."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from detection.data import jpeg_roundtrip, read_test
from detection.training import write_csv, write_json

from .canonical_augmentation import CORRELATION_VARIANT, CorrelationCNN
from .cnn import device_for, read_json, run_lock
from .codec_anchor import prepare_data, recenter
from .codec_canonical import CANONICAL, compare_views, transform, validation_halves
from .codec_final import profile_gain
from .codec_patch import encode
from .codec_response import PairedViews
from .codec_scale import restore_candidate
from .deep_fusion import comparison
from .night_train import CROP, check_time, image_patches, scores
from .overnight_training import TARGET_CORRECT, assess_export
from .source_holdout import ReleasedReference
from .spectrum_patch import QUALITY


RUNS = Path("lqy/runs")
BASE_CSV = RUNS / "resize-specialist/confirm-01/submission.csv"
ONLINE_CORRECT = 628
FIT_PROFILES = ("native", "jpeg83", "resize", "webp")
SCREEN = ("native", "jpeg76", "alternate_resize_jpeg", "webp")
CONFIRM = ("native", "jpeg83", "resize", "resize_webp")
WEIGHTS = (0.5, 1.0)


def grid_probe(rgb):
    short = min(rgb.size)
    nearest = 2 ** round(math.log2(short))
    factor = nearest / short
    if factor == 1 or not 0.76 <= factor <= 1.31:
        return rgb
    return rgb.resize(tuple(max(1, round(n * factor)) for n in rgb.size), Image.Resampling.LANCZOS)


def mild_probe(rgb):
    factor = 0.88 if min(rgb.size) >= 160 else 1.16
    return rgb.resize(tuple(max(1, round(n * factor)) for n in rgb.size), Image.Resampling.LANCZOS)


def render_response(item, profile, encoder, probe):
    with Image.open(item.path) as source:
        rgb = transform(source.convert("RGB"), profile)
    canonical = jpeg_roundtrip(probe(rgb), quality=CANONICAL, subsampling=2)
    return image_patches(canonical, QUALITY), image_patches(canonical, QUALITY, encoder=encoder)


def response_views(items, profile, table, deadline, *, probe=mild_probe):
    first = np.empty((len(items), 5, 3, CROP, CROP), dtype=np.uint8)
    second = np.empty_like(first)
    render = partial(render_response, profile=profile, encoder=partial(encode, table=table), probe=probe)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(items), 32):
            check_time(deadline)
            for i, (a, b) in enumerate(workers.map(render, items[start:start + 32]), start=start):
                first[i], second[i] = a, b
    return PairedViews(torch.from_numpy(first), torch.from_numpy(second))


def anchor_rows(samples, labels, full):
    anchors = []
    for label in (0, 1):
        subset = sorted((row for row in full.tolist() if int(labels[row]) == label),
                        key=lambda row: sha256(f"full-canonical-anchor:{samples[row].image_id}".encode()).digest())
        anchors.extend(subset[-320:])
    return anchors


def response_features(raw, perturbed):
    keep = np.asarray(raw).clip(0.001, 0.999)
    probe = np.asarray(perturbed).clip(0.001, 0.999)
    native = np.log(keep / (1 - keep))
    altered = np.log(probe / (1 - probe))
    change = altered - native
    return np.column_stack((native.T, altered.T, np.abs(change).T,
                            np.abs(native[0] - native[1]), np.abs(altered[0] - altered[1])))


def run(args, device, deadline):
    samples, labels, sizes, _, _, full, table, identity = prepare_data()
    selection, confirmation = validation_halves(samples, labels, sizes, len(full))
    reference = ReleasedReference(args, device, identity)
    detectors, thresholds = [], []
    for folder in (args.specialist, args.correlation):
        state = read_json(folder / "summary.json")
        if (state["split_sha256"] != identity or state["variant"] != CORRELATION_VARIANT
                or state["calibration_count"] != 640):
            raise ValueError("Expected both canonical detectors trained without the calibration pool")
        detectors.append(restore_candidate(folder / "full" / CORRELATION_VARIANT,
                                           CORRELATION_VARIANT, CorrelationCNN, device))
        thresholds.append(state["threshold"])
    probe = grid_probe if args.probe == "grid" else mild_probe
    report = {"split_sha256": identity, "baseline_online_accuracy": ONLINE_CORRECT / 1000, "probe": args.probe,
              "screen": {}, "confirmation": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    def predict(items, profile, *, fit_head=False):
        native, original, canonical, guard = compare_views(items, profile, table, deadline)
        rows = torch.arange(len(items))
        raw = [scores(models, canonical, rows, device, deadline) for models in detectors]
        previous = None
        if not fit_head:
            previous = 0.75 * reference.score(native, original, guard, rows, deadline) + 0.25 * recenter(raw[0], thresholds[0])
        del native, original, canonical
        response = response_views(items, profile, table, deadline, probe=probe)
        perturbed = [scores(models, response, rows, device, deadline) for models in detectors]
        return previous, response_features(raw, perturbed)

    anchors = anchor_rows(samples, labels, full)
    items = [samples[i] for i in anchors]
    fitted = []
    for profile in FIT_PROFILES:
        record(f"fit_head_{profile}")
        _, features = predict(items, profile, fit_head=True)
        fitted.append(features)
    head = make_pipeline(StandardScaler(), LogisticRegression(C=0.10, max_iter=500))
    head.fit(np.concatenate(fitted), np.tile(labels[anchors].numpy(), len(fitted)))
    report["fit_originals"] = len(anchors)

    def verify(indices, profiles, weights, key):
        examples = [samples[i] for i in indices]
        truth = labels[indices].numpy()
        paired = {weight: [] for weight in weights}
        for profile in profiles:
            record(f"{key}_{profile}")
            before, data = predict(examples, profile)
            probability = head.predict_proba(data)[:, 1]
            report[key][profile] = {}
            for weight in weights:
                after = (1 - weight) * before + weight * probability
                measured = comparison(truth, before, after)
                paired[weight].append((before, after))
                report[key][profile][str(weight)] = measured
                print(f"{key} {profile} {weight:g}: {measured['reference']['accuracy']:.2%} -> "
                      f"{measured['deep_fusion']['accuracy']:.2%}", flush=True)
        return {weight: profile_gain(truth, rows) for weight, rows in paired.items()}

    screen = verify(selection, SCREEN, WEIGHTS, "screen")
    report["screen_gain"] = screen
    eligible = [weight for weight in WEIGHTS if screen[weight]["mean_gain"] >= 0.025
                and screen[weight]["paired_95_percent"][0] > 0
                and min(r[str(weight)]["delta"]["accuracy"] for r in report["screen"].values()) >= -0.015]
    if not eligible:
        record("screen_fell_short")
        return
    weight = max(eligible, key=lambda w: screen[w]["mean_gain"])
    report["selected_weight"] = weight
    confirmed = verify(confirmation, CONFIRM, (weight,), "confirmation")[weight]
    report["confirmation_gain"] = confirmed
    if (confirmed["mean_gain"] < 0.025 or confirmed["paired_95_percent"][0] <= 0
            or min(r[str(weight)]["delta"]["accuracy"] for r in report["confirmation"].values()) < -0.01):
        record("confirmation_fell_short")
        return

    record("export")
    template = Path("ai-generated-image-detection-benchmark/sample_submission.csv")
    test = read_test(template, Path("dataset/dataset/test"))
    before, data = predict(test, "native")
    result = (1 - weight) * before + weight * head.predict_proba(data)[:, 1]
    path = args.output / "submission.csv"
    write_csv(path, ("image_id", "label"), ((item.image_id, int(score > 0.5))
                                           for item, score in zip(test, result, strict=True)))
    checks = assess_export(path, reference=BASE_CSV, known_correct=ONLINE_CORRECT)
    report["csv_checks"] = checks
    record("candidate_ready" if checks["best_possible_correct"] >= TARGET_CORRECT else "cannot_reach_target")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--probe", choices=("mild", "grid"), default="mild")
    parser.add_argument("--specialist", type=Path, default=RUNS / "resize-specialist/round-01")
    parser.add_argument("--correlation", type=Path, default=RUNS / "codec-canonical/correlation-full-01")
    parser.add_argument("--narrow-reference", type=Path, default=RUNS / "codec-final/upstream-trained")
    parser.add_argument("--near-reference", type=Path, default=RUNS / "codec-final/near-upstream")
    parser.add_argument("--night-reference", type=Path, default=RUNS / "night-train/2026-11-21")
    parser.add_argument("--spectrum-reference", type=Path, default=RUNS / "spectrum-patch/round-01")
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

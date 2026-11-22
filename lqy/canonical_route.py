"""Learn when a scratch JPEG detector improves on the released detector."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from detection.data import read_test
from detection.training import write_csv, write_json

from .cnn import device_for, read_json, run_lock
from .codec_anchor import prepare_data, recenter
from .codec_canonical import compare_views, transform
from .codec_final import profile_gain
from .codec_scale import restore_candidate
from .deep_fusion import comparison
from .night_train import scores
from .source_holdout import ReleasedReference, SourceCNN


FIT_PROFILES = ("native", "jpeg83", "resize", "webp")
VALIDATE_PROFILES = ("native", "jpeg76", "resize", "resize_jpeg", "webp")
CONFIDENCE = 0.60


def resize_evidence(rgb):
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    h, w = gray.shape
    y, x = max(0, (h - 320) // 2), max(0, (w - 320) // 2)
    gray = gray[y:y + 320, x:x + 320]
    signal = []
    for axis in (0, 1):
        line = np.abs(np.diff(gray, n=2, axis=axis)).mean(axis=1 - axis)
        expected = max(1e-3, float(line.mean())) if len(line) else 1e-3
        signal.append(np.log1p(expected))
        for period in range(3, 9):
            blocks = len(line) // period
            if blocks < 3:
                signal.append(0.0)
            else:
                means = line[:blocks * period].reshape(blocks, period).mean(axis=0)
                signal.append(float(means.std()) / expected)
    return signal


def features(items, profile, baseline, candidate):
    def feature(item):
        with Image.open(item.path) as source:
            return resize_evidence(transform(source.convert("RGB"), profile))

    with ThreadPoolExecutor(max_workers=3) as workers:
        images = np.asarray(list(workers.map(feature, items)))
    return np.column_stack((baseline - 0.5, candidate - 0.5, np.abs(baseline - 0.5),
                            np.abs(candidate - 0.5), images))


def labeled_split(items):
    fit, validation = [], []
    for label in (0, 1):
        rows = sorted((i for i, item in enumerate(items) if item.label == label),
                      key=lambda i: sha256(f"canonical-route-independent:{items[i].image_id}".encode()).digest())
        middle = len(rows) // 2
        fit.extend(rows[:middle])
        validation.extend(rows[middle:])
    return np.asarray(fit), np.asarray(validation)


def fit_gate(groups):
    data = []
    labels = []
    for baseline, candidate, cues, truth in groups:
        differs = (candidate > 0.5) != (baseline > 0.5)
        data.append(cues[differs])
        labels.append((candidate[differs] > 0.5) == truth[differs])
    x, y = np.concatenate(data), np.concatenate(labels)
    if len(np.unique(y)) != 2:
        raise ValueError("Both outcomes are required to fit the disagreement model")
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.40, max_iter=400)).fit(x, y), len(y)


def route(gate, baseline, candidate, cues):
    result = baseline.copy()
    rows = np.flatnonzero((candidate > 0.5) != (baseline > 0.5))
    if len(rows):
        chosen = rows[gate.predict_proba(cues[rows])[:, 1] >= CONFIDENCE]
        result[chosen] = candidate[chosen]
    return result


def run(args, device, deadline):
    samples, labels, _, _, _, training, table, identity = prepare_data()
    state = read_json(args.canonical / "summary.json")
    variant = state["variant"]
    if state["split_sha256"] != identity or variant != "constrained_rgb" or "threshold" not in state:
        raise ValueError("The router needs the completed matched canonical checkpoints")
    reference = ReleasedReference(args, device, identity)
    models = restore_candidate(args.canonical / "full" / variant, variant, SourceCNN, device)
    validation = samples[len(training):]
    truth = labels[len(training):].numpy()
    fit_rows, check_rows = labeled_split(validation)
    report = {"split_sha256": identity, "confidence": CONFIDENCE, "calibration_count": len(fit_rows),
              "validation_count": len(check_rows), "profiles": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    def evaluate(items, profile):
        original, codec, canonical, guard = compare_views(items, profile, table, deadline)
        positions = torch.arange(len(items))
        before = reference.score(original, codec, guard, positions, deadline)
        after = recenter(scores(models, canonical, positions, device, deadline), state["threshold"])
        return before, after, features(items, profile, before, after)

    record("fit")
    fit_items = [validation[i] for i in fit_rows]
    groups = [(*evaluate(fit_items, profile), truth[fit_rows]) for profile in FIT_PROFILES]
    gate, events = fit_gate(groups)
    report["fit_disagreements"] = events
    pairs = []
    check_items = [validation[i] for i in check_rows]
    for profile in VALIDATE_PROFILES:
        record(f"validate_{profile}")
        before, after, cues = evaluate(check_items, profile)
        result = route(gate, before, after, cues)
        pairs.append((before, result))
        measured = comparison(truth[check_rows], before, result)
        report["profiles"][profile] = measured
        print(f"{profile}: {measured['reference']['accuracy']:.2%} -> {measured['deep_fusion']['accuracy']:.2%}", flush=True)
    gain = profile_gain(truth[check_rows], pairs)
    novel_gain = profile_gain(truth[check_rows], [pair for profile, pair in zip(VALIDATE_PROFILES, pairs, strict=True)
                                                if profile not in FIT_PROFILES])
    report.update(aggregate=gain, different_encodings=novel_gain)
    if (gain["mean_gain"] < 0.025 or gain["paired_95_percent"][0] <= 0 or novel_gain["paired_95_percent"][0] <= 0
            or any(r["delta"]["accuracy"] < -0.015 for r in report["profiles"].values())):
        record("stopped_at_validation")
        return

    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected a 1,000-image submission")
    prediction = route(gate, *evaluate(test, "native"))
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((item.image_id, int(p > 0.5)) for item, p in zip(test, prediction, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--canonical", type=Path, default=Path("lqy/runs/codec-canonical/round-01"))
    parser.add_argument("--narrow-reference", type=Path, default=Path("lqy/runs/codec-final/upstream-trained"))
    parser.add_argument("--near-reference", type=Path, default=Path("lqy/runs/codec-final/near-upstream"))
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

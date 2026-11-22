"""Check expanded, matched-window training against the 62.8% submission."""

import argparse
from functools import partial
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from detection.data import read_test
from detection.training import write_csv, write_json

from .canonical_augmentation import CORRELATION_VARIANT, CorrelationCNN
from .cnn import device_for, read_json, run_lock
from .codec_anchor import prepare_data, recenter
from .codec_canonical import compare_views, transform, validation_halves
from .codec_final import profile_gain
from .codec_scale import restore_candidate
from .deep_fusion import comparison
from .matched_acquisition import CALIBRATION, MATCHED_VALIDATION, WINDOW_SIDE, cache_views, calibrated, matched_image
from .matched_acquisition import non_small_pairs, training_histories, training_profiles
from .night_train import SCREEN_EPOCHS, SEEDS, fit, scores
from .overnight_training import TARGET_CORRECT, assess_export
from .source_holdout import ReleasedReference
from .view_response import BASE_CSV, ONLINE_CORRECT, anchor_rows


RUNS = Path("lqy/runs")
SCREEN = ("native", "jpeg83", "resize", "matched_jpeg81", "webp")
CONFIRM = ("native", "jpeg76", "alternate_resize_jpeg", "matched_webp84", "resize_webp")
WEIGHTS = (0.35, 0.65, 1.0)


def run(args, device, deadline):
    samples, labels, sizes, _, _, full, table, identity = prepare_data()
    specialist_state = read_json(args.specialist / "summary.json")
    if specialist_state["split_sha256"] != identity or specialist_state["variant"] != CORRELATION_VARIANT:
        raise ValueError("The confirmed submission and the candidate require the same labeled split")
    selection, confirmation = validation_halves(samples, labels, sizes, len(full))
    anchors = anchor_rows(samples, labels, full)
    reserved = set(anchors)
    ordered = [i for label in (0, 1) for i in full.tolist() if i not in reserved and int(labels[i]) == label]
    training = non_small_pairs(torch.tensor(ordered), sizes)
    anchors = non_small_pairs(torch.tensor(anchors), sizes)
    report = {"split_sha256": identity, "base_online_accuracy": ONLINE_CORRECT / 1000,
              "training_count": len(training), "calibration_count": len(anchors), "screen": {}, "confirmation": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("training")
    items = [samples[i] for i in training]
    print(f"Training on {len(items)} labeled images with matched pixel windows", flush=True)
    primary, second = training_profiles(labels[training])
    one, two = cache_views(items, primary, table, deadline, crops=[0, 1, 4], native_window=True,
                           initial_profiles=training_histories(items), alternate_profiles=second)
    models = [fit(args.output / "full" / CORRELATION_VARIANT / f"s{seed}", CORRELATION_VARIANT, seed,
                  SCREEN_EPOCHS, one, labels[training], torch.arange(len(items)), device, deadline,
                  model_factory=CorrelationCNN, paired_data=two) for seed in SEEDS]
    del one, two

    record("calibration")
    items = [samples[i] for i in anchors]
    pictures = cache_views(items, [CALIBRATION] * len(items), table, deadline,
                           crops=[0, 1, 2, 3, 4], native_window=True)
    threshold = calibrated(models, pictures, labels[anchors], device, deadline)
    report["threshold"] = threshold
    del pictures
    released = ReleasedReference(args, device, identity)
    specialist = restore_candidate(args.specialist / "full" / CORRELATION_VARIANT,
                                   CORRELATION_VARIANT, CorrelationCNN, device)

    def predict(items, profile):
        input_profile = profile.removeprefix("matched_")
        custom = input_profile in MATCHED_VALIDATION and profile.startswith("matched_")
        override = partial(matched_image, profile=MATCHED_VALIDATION[input_profile]) if custom else None
        standard = "native" if custom else profile
        original, codec, canonical, guard = compare_views(items, standard, table, deadline, input_transform=override)
        indexes = torch.arange(len(items))
        reference = released.score(original, codec, guard, indexes, deadline)
        resize = recenter(scores(specialist, canonical, indexes, device, deadline), specialist_state["threshold"])
        before = 0.75 * reference + 0.25 * resize
        del original, codec, canonical
        eligible = []
        for index, item in enumerate(items):
            with Image.open(item.path) as image:
                if min(image.size) >= WINDOW_SIDE:
                    eligible.append(index)
        after = before.copy()
        if eligible:
            subset = [items[i] for i in eligible]
            processing = override or partial(transform, profile=standard)
            data = cache_views(subset, [CALIBRATION] * len(subset), table, deadline, crops=[0, 1, 2, 3, 4],
                               native_window=True, input_transform=processing)
            after[eligible] = recenter(scores(models, data, torch.arange(len(subset)), device, deadline), threshold)
        return before, after

    def verify(rows, profiles, choices, section):
        items = [samples[i] for i in rows]
        truth = labels[rows].numpy()
        pairs = {weight: [] for weight in choices}
        for profile in profiles:
            record(f"{section}_{profile}")
            before, proposal = predict(items, profile)
            report[section][profile] = {}
            for weight in choices:
                after = (1 - weight) * before + weight * proposal
                measurement = comparison(truth, before, after)
                pairs[weight].append((before, after))
                report[section][profile][str(weight)] = measurement
                print(f"{section} {profile} {weight:g}: {measurement['reference']['accuracy']:.2%} -> "
                      f"{measurement['deep_fusion']['accuracy']:.2%}", flush=True)
        return {weight: profile_gain(truth, bundle) for weight, bundle in pairs.items()}

    screen = verify(selection, SCREEN, WEIGHTS, "screen")
    report["screen_gain"] = screen
    eligible = [weight for weight in WEIGHTS if screen[weight]["mean_gain"] >= 0.035
                and screen[weight]["paired_95_percent"][0] > 0
                and min(r[str(weight)]["delta"]["accuracy"] for r in report["screen"].values()) >= -0.01]
    if not eligible:
        record("screen_fell_short")
        return
    weight = max(eligible, key=lambda value: screen[value]["mean_gain"])
    report["selected_weight"] = weight
    confirm = verify(confirmation, CONFIRM, (weight,), "confirmation")[weight]
    report["confirmation_gain"] = confirm
    if (confirm["mean_gain"] < 0.035 or confirm["paired_95_percent"][0] <= 0
            or min(r[str(weight)]["delta"]["accuracy"] for r in report["confirmation"].values()) < -0.01):
        record("confirmation_fell_short")
        return

    record("export")
    template = Path("ai-generated-image-detection-benchmark/sample_submission.csv")
    test = read_test(template, Path("dataset/dataset/test"))
    before, proposal = predict(test, "native")
    after = (1 - weight) * before + weight * proposal
    path = args.output / "submission.csv"
    write_csv(path, ("image_id", "label"), ((item.image_id, int(p > 0.5)) for item, p in zip(test, after, strict=True)))
    checks = assess_export(path, reference=BASE_CSV, known_correct=ONLINE_CORRECT)
    report["csv_checks"] = checks
    record("candidate_ready" if checks["best_possible_correct"] >= TARGET_CORRECT else "cannot_reach_target")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--specialist", type=Path, default=RUNS / "resize-specialist/round-01")
    parser.add_argument("--narrow-reference", type=Path, default=RUNS / "codec-final/upstream-trained")
    parser.add_argument("--near-reference", type=Path, default=RUNS / "codec-final/near-upstream")
    parser.add_argument("--night-reference", type=Path, default=RUNS / "night-train/2026-11-21")
    parser.add_argument("--spectrum-reference", type=Path, default=RUNS / "spectrum-patch/round-01")
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

"""Train and verify a canonical JPEG detector with balanced, excluded calibration data."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
import io
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from detection.data import jpeg_roundtrip, read_test
from detection.training import write_csv, write_json

from .cnn import device_for, read_json, run_lock
from .codec_anchor import cutoff, prepare_data, recenter
from .codec_final import cache_stress, profile_gain
from .codec_invariant import training_views
from .codec_patch import encode
from .codec_response import PairedViews
from .codec_route import use_reference
from .codec_scale import HYBRID, HybridResponseCNN
from .deep_fusion import comparison
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, check_time, fit, image_patches, scores
from .source_holdout import ReleasedReference, SourceCNN, source_folds
from .spectrum_patch import QUALITY


CANONICAL = 82
CALIBRATION_PER_CLASS = 320
WEIGHTS = (0.5, 1.0)
SCREEN = ("native", "jpeg83", "resize", "webp")
CONFIRM = ("jpeg76", "resize_jpeg")


def balanced_cutoff(probabilities, labels):
    values = np.asarray(probabilities)
    cutoffs = np.r_[1e-6, np.unique(values).clip(1e-6, 1 - 1e-6), 1 - 1e-6]
    accuracy = ((values[:, None] > cutoffs) == labels[:, None]).mean(axis=0)
    rank = np.lexsort((np.abs(cutoffs - 0.5), -accuracy))
    return float(cutoffs[rank[0]])


def calibration_rules(probabilities, labels):
    balanced = balanced_cutoff(probabilities, labels)
    photo = cutoff(probabilities[labels == 0], 0.75)
    odds = np.sqrt(photo / (1 - photo) * balanced / (1 - balanced))
    return {"both_class": balanced, "photo_q82": photo,
            "guarded": max(photo, balanced), "logit_midpoint": float(odds / (1 + odds))}


def validation_halves(samples, labels, sizes, training_count):
    seen = {i for _, _, check in source_folds(samples, labels, sizes, training_count).values() for i in check.tolist()}
    confirmation = []
    for label in (0, 1):
        rows = [i for i in range(training_count, len(samples)) if labels[i] == label and i not in seen]
        if len(rows) < 400:
            raise ValueError("Insufficient independent validation samples")
        rows.sort(key=lambda i: sha256(f"canonical-confirmation:{samples[i].image_id}".encode()).digest())
        confirmation.extend(rows[:400])
    reserved = set(confirmation)
    selection = [i for i in range(training_count, len(samples)) if i not in reserved]
    return selection, confirmation


def transform(rgb, profile):
    if profile in ("resize", "resize_jpeg", "alternate_resize_jpeg", "resize_webp"):
        short = min(rgb.size)
        if profile == "alternate_resize_jpeg":
            scale, kernel = (0.84 if short >= 160 else 1.27), Image.Resampling.LANCZOS
        elif profile == "resize_webp":
            scale, kernel = (0.86 if short >= 160 else 1.24), Image.Resampling.BILINEAR
        else:
            scale, kernel = (0.8 if short >= 160 else 1.2), Image.Resampling.BICUBIC
        rgb = rgb.resize(tuple(max(1, round(size * scale)) for size in rgb.size), kernel)
    if profile in ("jpeg83", "jpeg76", "resize_jpeg", "alternate_resize_jpeg"):
        encoding = {"jpeg83": (83, 0), "jpeg76": (76, 1), "resize_jpeg": (88, 2),
                    "alternate_resize_jpeg": (79, 0)}
        quality, subsampling = encoding[profile]
        rgb = jpeg_roundtrip(rgb, quality=quality, subsampling=subsampling)
    if profile in ("webp", "resize_webp"):
        with io.BytesIO() as stream:
            rgb.save(stream, format="WEBP", quality=85 if profile == "webp" else 74, method=1)
            stream.seek(0)
            with Image.open(stream) as decoded:
                rgb = decoded.convert("RGB")
    return rgb


def compare_views(items, profile, table, deadline, *, candidate_transform=None, input_transform=None):
    arrays = [np.empty((len(items), 5, 3, CROP, CROP), dtype=np.uint8) for _ in range(4)]
    guarded = np.empty(len(items), dtype=bool)
    encoder = partial(encode, table=table)

    def render(sample):
        with Image.open(sample.path) as source:
            rgb = transform(source.convert("RGB"), profile)
        if input_transform is not None:
            rgb = input_transform(rgb)
        candidate_rgb = rgb if candidate_transform is None else candidate_transform(rgb)
        standardized = jpeg_roundtrip(candidate_rgb, quality=CANONICAL, subsampling=2)
        return (image_patches(rgb, QUALITY), image_patches(rgb, QUALITY, encoder=encoder),
                image_patches(standardized, QUALITY), image_patches(standardized, QUALITY, encoder=encoder),
                use_reference(rgb))

    print(f"Preparing {len(items)} images for {profile}", flush=True)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(items), 32):
            check_time(deadline)
            for i, result in enumerate(workers.map(render, items[start:start + 32]), start=start):
                for array, data in zip(arrays, result[:4], strict=True):
                    array[i] = data
                guarded[i] = result[4]
    original, alternate, canonical, canonical_alt = map(torch.from_numpy, arrays)
    return original, PairedViews(original, alternate), PairedViews(canonical, canonical_alt), guarded


def run(args, device, deadline, *, training_loader=None, augmentation_check=None,
        candidate_transform=None, calibration_loader=None, model_variant=None,
        model_factory=None, paired_objective=None, candidate_adapter=None, screen_weights=WEIGHTS,
        confirm_profiles=CONFIRM):
    pilot = read_json(args.calibration_reference)
    samples, labels, sizes, _, _, full, table, identity = prepare_data()
    if pilot["split_sha256"] != identity or len(pilot["cohorts"]) != 2:
        raise ValueError("Both labeled source checks must finish on the same split")
    if "selected" in pilot:
        variant, rule = pilot["selected"]["variant"], pilot["selected"]["rule"]
        accuracy = [group["models"][variant]["rules"][rule]["accuracy"] for group in pilot["cohorts"].values()]
        rankings = {variant: (min(accuracy), float(np.mean(accuracy)))}
    else:
        rule = "both_class"
        rankings = {}
        for name in (HYBRID, "constrained_residual"):
            accuracy = [group["models"][name]["both_class"]["accuracy"] for group in pilot["cohorts"].values()]
            rankings[name] = min(accuracy), float(np.mean(accuracy))
        variant = max(rankings, key=rankings.get)
    factory = model_factory or (HybridResponseCNN if variant == HYBRID else SourceCNN)
    training_variant = model_variant or variant
    report = {"variant": training_variant, "split_sha256": identity, "source_ranking": rankings, "canonical_quality": CANONICAL,
              "calibration_rule": rule,
              "screen": {}, "confirmation": {}}
    if augmentation_check is not None:
        report["augmentation_check"] = augmentation_check

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    if rankings[variant][0] < 0.70 or rankings[variant][1] < 0.78:
        record("stopped_at_source_check")
        return

    training, anchor = [], []
    for label in (0, 1):
        indices = sorted((i for i in full.tolist() if int(labels[i]) == label),
                         key=lambda i: sha256(f"full-canonical-anchor:{samples[i].image_id}".encode()).digest())
        if len(indices) <= CALIBRATION_PER_CLASS:
            raise ValueError("Insufficient independent calibration examples")
        training.extend(indices[:-CALIBRATION_PER_CLASS])
        anchor.extend(indices[-CALIBRATION_PER_CLASS:])
    report["train_count"], report["calibration_count"] = len(training), len(anchor)
    record("training")
    items = [samples[i] for i in training]
    if training_loader is None:
        native, paired, _ = training_views(items, table, deadline)
    else:
        native, paired = training_loader(items, table, deadline)
    train_rows = torch.arange(len(training))
    nets = [fit(args.output / "full" / training_variant / f"s{seed}", training_variant,
                seed, SCREEN_EPOCHS, native, labels[training], train_rows, device, deadline,
                model_factory=factory, paired_data=paired, paired_objective=paired_objective) for seed in SEEDS]
    del native, paired
    items = [samples[i] for i in anchor]
    if calibration_loader is None:
        _, calibration = cache_stress(items, CANONICAL, table, deadline)
    else:
        calibration = calibration_loader(items, table, deadline)
    threshold = calibration_rules(scores(nets, calibration, torch.arange(len(anchor)), device, deadline), labels[anchor].numpy())[rule]
    report["threshold"] = threshold
    del calibration

    previous = ReleasedReference(args, device, identity)
    selection, confirmation = validation_halves(samples, labels, sizes, len(full))
    report["selection_validation_count"] = len(selection)
    report["confirmation_validation_count"] = len(confirmation)

    def predict(items, profile):
        native, original, canonical, guarded = compare_views(
            items, profile, table, deadline, candidate_transform=candidate_transform)
        rows = torch.arange(len(items))
        before = previous.score(native, original, guarded, rows, deadline)
        if candidate_adapter is not None:
            canonical = candidate_adapter(canonical, items, profile, table, deadline)
        after = recenter(scores(nets, canonical, rows, device, deadline), threshold)
        return before, after

    def verify(profiles, weights, key, indices):
        validation = [samples[i] for i in indices]
        truth = labels[indices].numpy()
        paired = {weight: [] for weight in weights}
        for profile in profiles:
            record(f"validate_{profile}")
            before, after = predict(validation, profile)
            report[key][profile] = {}
            for weight in weights:
                candidate = (1 - weight) * before + weight * after
                measured = comparison(truth, before, candidate)
                report[key][profile][str(weight)] = measured
                paired[weight].append((before, candidate))
                print(f"{profile} w={weight:g}: {measured['reference']['accuracy']:.2%} -> "
                      f"{measured['deep_fusion']['accuracy']:.2%}", flush=True)
            record(f"done_{profile}")
        return {weight: profile_gain(truth, rows) for weight, rows in paired.items()}

    result = verify(SCREEN, screen_weights, "screen", selection)
    report["screen_gain"] = result
    allowed = [weight for weight, gain in result.items() if gain["mean_gain"] >= 0.025
               and gain["paired_95_percent"][0] > 0
               and all(row[str(weight)]["delta"]["accuracy"] >= -0.015 for row in report["screen"].values())]
    if not allowed:
        record("stopped_at_screen")
        return
    weight = max(allowed, key=lambda item: result[item]["mean_gain"])
    report["selected_weight"] = weight
    confirmed = verify(confirm_profiles, (weight,), "confirmation", confirmation)[weight]
    report["confirmation_gain"] = confirmed
    if (confirmed["paired_95_percent"][0] <= 0 or
            any(item[str(weight)]["delta"]["accuracy"] < -0.01 for item in report["confirmation"].values())):
        record("stopped_at_confirmation")
        return

    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected a 1,000-image submission")
    before, after = predict(test, "native")
    combined = (1 - weight) * before + weight * after
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((item.image_id, int(value > 0.5)) for item, value in zip(test, combined, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--calibration-reference", type=Path, default=Path("lqy/runs/source-holdout/canonical-calibration.json"))
    parser.add_argument("--narrow-reference", type=Path, default=Path("lqy/runs/codec-final/upstream-trained"))
    parser.add_argument("--near-reference", type=Path, default=Path("lqy/runs/codec-final/near-upstream"))
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

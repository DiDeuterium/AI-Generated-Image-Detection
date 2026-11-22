"""Verify a JPEG-aware blend whose decision uses decoded pixels only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, partial
import io
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from detection.data import jpeg_roundtrip, read_test
from detection.training import write_csv, write_json

from .cnn import device_for, read_json, run_lock
from .codec_anchor import prepare_data, recenter
from .codec_final import profile_gain
from .codec_patch import encode
from .codec_response import PairedViews
from .codec_scale import HYBRID, HybridResponseCNN, restore_candidate
from .deep_fusion import comparison
from .night_train import CROP, check_time, image_patches, scores
from .phase_patch import full_reference
from .spectrum_patch import QUALITY


PROFILES = (None, 85, 70, 92, 67, 73)
CONFIRM = (72, 74)
STANDALONE_CHECK = (71, 75, 76)
WEIGHT = 0.35
NEAR_WEIGHT = 0.50
GUARD = (71, 76)
QUALITY_LEVELS = np.arange(50, 99)
AXIS = np.arange(8)
BASIS = (np.cos(np.pi * AXIS[:, None] * (2 * AXIS + 1) / 16) * 0.5).astype(np.float32)
BASIS[0] /= np.sqrt(2)
FREQUENCIES = (AXIS[:, None] + AXIS).ravel()
FREQUENCY_INDICES = np.flatnonzero((FREQUENCIES >= 2) & (FREQUENCIES <= 9))


@lru_cache(maxsize=1)
def quantization_codebook():
    tables = []
    blank = Image.new("L", (8, 8))
    for quality in QUALITY_LEVELS:
        with io.BytesIO() as stream:
            blank.save(stream, format="JPEG", quality=int(quality))
            stream.seek(0)
            with Image.open(stream) as decoded:
                tables.append(np.asarray(decoded.quantization[0], dtype=np.float32)[FREQUENCY_INDICES])
    return np.asarray(tables)[:, None, :]


def use_reference(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    height, width = (size // 8 * 8 for size in gray.shape)
    if min(height, width) < 8:
        return False
    blocks = gray[:height, :width].reshape(height // 8, 8, width // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    blocks = blocks[np.linspace(0, len(blocks) - 1, min(140, len(blocks)), dtype=int)]
    coefficients = np.abs((BASIS @ (blocks - 128) @ BASIS.T).reshape(-1, 64)[:, FREQUENCY_INDICES])
    tables = quantization_codebook()
    eligible = (tables >= 3) & (coefficients > tables * 0.8) & (coefficients > 2)
    weights = np.where(eligible, np.log1p(tables), 0)
    # Fractional DCT phase prevents fine quantizers from winning merely
    # because their nearest grid point is always close in absolute units.
    phase = np.cos(2 * np.pi * coefficients / tables)
    agreement = (phase * weights).sum(axis=(1, 2)) / np.maximum(weights.sum(axis=(1, 2)), 1)
    agreement *= np.minimum(1, eligible.sum(axis=(1, 2)) / 80)
    best = int(agreement.argmax())
    return bool(GUARD[0] <= QUALITY_LEVELS[best] <= GUARD[1] and agreement[best] >= 0.4)


def prepare(sample, quality, table, subsampling):
    with Image.open(sample.path) as image:
        rgb = image.convert("RGB")
    if quality is not None:
        rgb = jpeg_roundtrip(rgb, quality=quality, subsampling=subsampling)
    reference_only = use_reference(rgb)
    original = image_patches(rgb, QUALITY)
    alternate = image_patches(rgb, QUALITY, encoder=partial(encode, table=table))
    return original, alternate, reference_only


def cache_routed(items, quality, table, deadline, *, subsampling=2):
    print(f"Scoring {len(items)} images with upstream quality {quality}", flush=True)
    original = np.empty((len(items), 5, 3, CROP, CROP), dtype=np.uint8)
    alternate = np.empty_like(original)
    fallback = np.empty(len(items), dtype=bool)
    transform = partial(prepare, quality=quality, table=table, subsampling=subsampling)
    quantization_codebook()
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(items), 32):
            check_time(deadline)
            for i, (first, second, guarded) in enumerate(workers.map(transform, items[start:start + 32]), start=start):
                original[i], alternate[i], fallback[i] = first, second, guarded
    views = torch.from_numpy(original)
    return views, PairedViews(views, torch.from_numpy(alternate)), fallback


def run(args, device, deadline):
    model_summary = read_json(args.model_reference / "summary.json")
    blend_summary = read_json(args.blend_reference / "summary.json")
    if blend_summary.get("selected_weight") != WEIGHT:
        raise ValueError("The original blend must be selected before applying the pixel guard")
    samples, labels, _, _, _, train, table, identity = prepare_data()
    if model_summary["split_sha256"] != identity or blend_summary["split_sha256"] != identity:
        raise ValueError("Stored candidates use a different labeled split")
    threshold = model_summary["threshold"]
    models = restore_candidate(args.model_reference / "full" / HYBRID, HYBRID, HybridResponseCNN, device)
    specialist, specialist_threshold = None, None
    if args.near_reference is not None:
        near_summary = read_json(args.near_reference / "summary.json")
        if near_summary["stage"] != "ready_for_route" or near_summary["split_sha256"] != identity:
            raise ValueError("The near-Q75 checkpoint must use the same training and calibration data")
        specialist_threshold = near_summary["threshold"]
        specialist = restore_candidate(args.near_reference / "full" / HYBRID, HYBRID, HybridResponseCNN, device)
    if args.standalone_near and specialist is None:
        raise ValueError("Standalone verification requires the near-Q75 checkpoint")
    near_weight = 1.0 if args.standalone_near else NEAR_WEIGHT
    val = [samples[i] for i in range(len(train), len(samples))]
    truth = labels[len(train):].numpy()
    report = {"split_sha256": identity, "blend_weight": WEIGHT, "guard_range": GUARD, "profiles": {}, "confirmation": {},
              "near_weight": near_weight if specialist is not None else 0.0}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    def predict(items, quality, subsampling=2):
        native, paired, fallback = cache_routed(items, quality, table, deadline, subsampling=subsampling)
        rows = torch.arange(len(items))
        reference = full_reference(args, native, rows, device, deadline)
        alternative = recenter(scores(models, paired, rows, device, deadline), threshold)
        result = np.where(fallback, reference, (1 - WEIGHT) * reference + WEIGHT * alternative)
        incumbent = result.copy()
        if specialist is not None and fallback.any():
            near_rows = rows[torch.from_numpy(fallback)]
            near = recenter(scores(specialist, paired, near_rows, device, deadline), specialist_threshold)
            incumbent[fallback] = (1 - NEAR_WEIGHT) * reference[fallback] + NEAR_WEIGHT * near
            result[fallback] = (1 - near_weight) * reference[fallback] + near_weight * near
        return reference, result, fallback, incumbent

    def verify(qualities, key, subsampling=2):
        pairs = []
        for quality in qualities:
            name = "native" if quality is None else f"q{quality}"
            record(f"validate_{name}")
            reference, result, fallback, _ = predict(val, quality, subsampling)
            entry = comparison(truth, reference, result)
            pairs.append((reference, result))
            report[key][name] = {"fallback_count": int(fallback.sum()), **entry}
            print(f"{name}: accuracy {entry['reference']['accuracy']:.3%} -> "
                  f"{entry['deep_fusion']['accuracy']:.3%}, guarded {int(fallback.sum())}/{len(fallback)}", flush=True)
        return profile_gain(truth, pairs)

    if args.standalone_near:
        report["standalone_check"] = {}
        pairs = []
        for quality in STANDALONE_CHECK:
            record(f"standalone_q{quality}")
            _, alternative, fallback, incumbent = predict(val, quality, subsampling=1)
            measured = comparison(truth, incumbent, alternative)
            report["standalone_check"][f"q{quality}"] = {"guarded": int(fallback.sum()), **measured}
            pairs.append((incumbent, alternative))
            print(f"Q{quality} standalone vs equal blend: {measured['reference']['accuracy']:.3%} -> "
                  f"{measured['deep_fusion']['accuracy']:.3%}", flush=True)
        evidence = profile_gain(truth, pairs)
        report["standalone_gain"] = evidence
        if (evidence["mean_gain"] < 0.01 or evidence["paired_95_percent"][0] <= 0
                or any(item["delta"]["accuracy"] < -0.015 for item in report["standalone_check"].values())):
            record("stopped_at_standalone_check")
            return

    main = verify(PROFILES, "profiles")
    report["aggregate"] = main
    if (main["mean_gain"] < 0.03 or main["paired_95_percent"][0] <= 0
            or any(entry["delta"]["accuracy"] < -0.015 or entry["deep_fusion"]["accuracy"] < 0.75
                   for entry in report["profiles"].values())):
        record("stopped_at_validation")
        return
    confirmation_subsampling = 2 if specialist is None else 0
    report["confirmation_subsampling"] = confirmation_subsampling
    report["confirmation_summary"] = verify(CONFIRM, "confirmation", subsampling=confirmation_subsampling)
    if any(entry["delta"]["accuracy"] < -0.01 or entry["deep_fusion"]["accuracy"] < 0.75
           for entry in report["confirmation"].values()):
        record("stopped_at_confirmation")
        return

    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    _, predictions, _, _ = predict(test, None)
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(score > 0.5)) for sample, score in zip(test, predictions, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--model-reference", type=Path, default=Path("lqy/runs/codec-final/upstream-trained"))
    parser.add_argument("--blend-reference", type=Path, default=Path("lqy/runs/codec-final/fixed-blend"))
    parser.add_argument("--near-reference", type=Path)
    parser.add_argument("--standalone-near", action="store_true", help="First compare a standalone specialist on new labeled profiles")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

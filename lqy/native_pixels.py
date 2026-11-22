"""Separate the effects of blur, scaling, and re-encoding on fixed native crops."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter
import torch

from detection.data import jpeg_roundtrip
from detection.metrics import binary_metrics
from detection.training import write_json

from .canonical_augmentation import CORRELATION_VARIANT, CorrelationCNN, calibration_indices
from .cnn import device_for, read_json, run_lock
from .codec_anchor import prepare_data, recenter
from .codec_canonical import balanced_cutoff
from .codec_patch import encode
from .codec_response import PairedViews
from .codec_scale import restore_candidate
from .deep_fusion import comparison
from .matched_acquisition import WINDOW_POSITIONS
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, check_time, fit, scores
from .source_holdout import source_folds, source_gain
from .spectrum_patch import QUALITY


RUNS = Path("lqy/runs")
CONTROLS = ("native", "blur", "resize", "jpeg83", "combined")
VERIFY = ("native", "jpeg76", "resize_roundtrip")


def operate(patch, profile):
    if profile in ("blur", "combined"):
        patch = patch.filter(ImageFilter.GaussianBlur(radius=0.55))
    if profile in ("resize", "combined", "resize_roundtrip"):
        side = 158 if profile == "resize_roundtrip" else 176
        patch = patch.resize((side, side), Image.Resampling.BICUBIC)
        patch = patch.resize((CROP, CROP), Image.Resampling.LANCZOS)
    if profile in ("jpeg83", "combined", "jpeg76", "jpeg91"):
        quality = {"jpeg83": 83, "combined": 83, "jpeg76": 76, "jpeg91": 91}[profile]
        patch = jpeg_roundtrip(patch, quality=quality, subsampling=2)
    return patch


def paired_rows(rows, sizes):
    a, b = rows.chunk(2)
    keep = torch.tensor([min(*sizes[i], *sizes[j]) >= CROP
                         for i, j in zip(a.tolist(), b.tolist(), strict=True)], dtype=torch.bool)
    return torch.cat((a[keep], b[keep]))


def cache(items, table, deadline, *, profiles, crops, alternate=None):
    groups = (profiles,) if alternate is None else (profiles, alternate)
    settings = list(zip(*groups, strict=True))
    if len(settings) != len(items):
        raise ValueError("A codec setup is required for every image")
    primary = np.empty((len(groups), len(items), len(crops), 3, CROP, CROP), dtype=np.uint8)
    secondary = np.empty_like(primary)
    encoder = partial(encode, table=table)

    def render(item, formats):
        with Image.open(item.path) as source:
            rgb = source.convert("RGB")
        width, height = rgb.size
        if min(width, height) < CROP:
            raise ValueError("A complete 128px native crop is required")
        windows = []
        for index in crops:
            x, y = WINDOW_POSITIONS[index]
            left, top = round((width - CROP) * x), round((height - CROP) * y)
            windows.append(rgb.crop((left, top, left + CROP, top + CROP)))
        return [[operate(window, profile) for window in windows] for profile in formats]

    def pack(item, formats):
        altered = render(item, formats)
        return [
            [(np.asarray(jpeg_roundtrip(view, quality=QUALITY, subsampling=2)).transpose(2, 0, 1),
              np.asarray(encoder(view)).transpose(2, 0, 1)) for view in views]
            for views in altered
        ]

    with ThreadPoolExecutor(max_workers=4) as workers:
        for first in range(0, len(items), 24):
            check_time(deadline)
            stop = first + 24
            for row, result in enumerate(workers.map(pack, items[first:stop], settings[first:stop]), start=first):
                for view, windows in enumerate(result):
                    for crop_index, (a, b) in enumerate(windows):
                        primary[view, row, crop_index] = a
                        secondary[view, row, crop_index] = b
    outputs = tuple(PairedViews(torch.from_numpy(a), torch.from_numpy(b))
                    for a, b in zip(primary, secondary, strict=True))
    return outputs if alternate is not None else outputs[0]


def run(args, device, deadline):
    samples, labels, sizes, _, _, full, table, identity = prepare_data()
    controls = read_json(args.reference / "summary.json")
    if controls["split_sha256"] != identity or controls["variant"] != CORRELATION_VARIANT:
        raise ValueError("Scratch reference does not use the current training split")
    state = {"split_sha256": identity, "crop_side": CROP, "mode": args.mode, "cohorts": {}}

    def record(stage):
        state["stage"] = stage
        write_json(args.output / "summary.json", state)

    def estimate(rows, setups, models):
        items = [samples[i] for i in rows.tolist()]
        data = cache(items, table, deadline, profiles=setups, crops=[0, 1, 2, 3, 4])
        return scores(models, data, torch.arange(len(rows)), device, deadline)

    all_pairs = []
    for name, (original_training, photographs, validation) in source_folds(samples, labels, sizes, len(full)).items():
        record(f"working_{name}")
        anchors = calibration_indices(samples, labels, sizes, len(full), name, original_training, photographs)
        training, anchors, validation = (paired_rows(rows, sizes) for rows in (original_training, anchors, validation))
        reference = restore_candidate(args.reference / name, CORRELATION_VARIANT, CorrelationCNN, device)
        true = labels[validation].numpy()
        report = {"train_count": len(training), "anchor_count": len(anchors), "validation_count": len(validation),
                  "profiles": {}}
        if args.mode == "audit":
            cutoff = balanced_cutoff(estimate(anchors, ["native"] * len(anchors), reference), labels[anchors].numpy())
            for setup in CONTROLS:
                p = estimate(validation, [setup] * len(validation), reference)
                info = binary_metrics(true.tolist(), p > cutoff, p)
                report["profiles"][setup] = info
                print(f"{name} {setup}: {info['accuracy']:.2%}, AUROC {info['auroc']:.4f}", flush=True)
        else:
            count = len(training) // 2
            setup_a = (["native", "jpeg91"] * ((count + 1) // 2))[:count] * 2
            setup_b = (["jpeg83", "native"] * ((count + 1) // 2))[:count] * 2
            items = [samples[i] for i in training.tolist()]
            print(f"{name}: preparing {len(items)} labeled native crops", flush=True)
            data, other = cache(items, table, deadline, profiles=setup_a, alternate=setup_b, crops=[0, 1, 4])
            models = [fit(args.output / name / f"s{seed}", CORRELATION_VARIANT, seed, SCREEN_EPOCHS,
                          data, labels[training], torch.arange(len(training)), device, deadline,
                          model_factory=CorrelationCNN, paired_data=other) for seed in SEEDS]
            del data, other
            ref_cut = balanced_cutoff(estimate(anchors, ["native"] * len(anchors), reference), labels[anchors].numpy())
            new_cut = balanced_cutoff(estimate(anchors, ["native"] * len(anchors), models), labels[anchors].numpy())
            report["threshold"] = new_cut
            pairs = []
            for setup in VERIFY:
                p = recenter(estimate(validation, [setup] * len(validation), reference), ref_cut)
                q = recenter(estimate(validation, [setup] * len(validation), models), new_cut)
                info = comparison(true, p, q)
                pairs.append((p, q))
                report["profiles"][setup] = info
                print(f"{name} {setup}: {info['reference']['accuracy']:.2%} -> "
                      f"{info['deep_fusion']['accuracy']:.2%}", flush=True)
            all_pairs.append((true, pairs))
        state["cohorts"][name] = report
        record(f"verified_{name}")
    if all_pairs:
        state["source_gain"] = source_gain(all_pairs)
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("audit", "train"), required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--reference", type=Path, default=RUNS / "canonical-augmentation/correlation-01")
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

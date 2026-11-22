"""Test smaller native crops where a 128px image has only one distinct 128px crop."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

from detection.data import jpeg_roundtrip, read_test
from detection.training import load_checkpoint, write_csv, write_json

from .cnn import device_for, read_json, run_lock
from .codec_anchor import cutoff, exclude_anchor, prepare_data, recenter, view_pair
from .codec_patch import encode
from .codec_response import VARIANT, CodecResponseCNN, train_variant
from .deep_fusion import MIN_CONFIRM_GAIN, MIN_SCREEN_GAIN, bootstrap_mean_gain, combined_reference, comparison, spectral_views
from .group_patch import restore_ensemble
from .heldout_blend import restore
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, cache, check_time, scores
from .spectrum_patch import QUALITY, VARIANT as SPEC_VARIANT, SpectrumPatchCNN


CROP_SIZE = 64
QUANTILE = 0.75
SCREEN = (128, 512, 1024)
CONFIRM = 256
NORMALIZED = "codec_response_normalized"
HYBRID = "codec_response_hybrid"


class ThumbnailTrainingViews:
    def __init__(self, original, thumbnails, trained_rows, sample_count):
        self.original = original
        self.thumbnails = thumbnails
        self.position = torch.full((sample_count,), -1, dtype=torch.int64)
        self.position[trained_rows] = torch.arange(len(trained_rows))

    def __getitem__(self, rows):
        original = self.original[rows]
        lookup = self.position[rows]
        alternate = (lookup >= 0) & (torch.rand(len(rows)) < 0.5)
        original[alternate] = self.thumbnails[lookup[alternate], None].expand(-1, original.shape[1], -1, -1, -1)
        return original


def thumbnail(sample, table):
    with Image.open(sample.path) as image:
        source = ImageOps.fit(image.convert("RGB"), (CROP, CROP), Image.Resampling.LANCZOS)
    standard = jpeg_roundtrip(source, quality=QUALITY, subsampling=2)
    alternate = encode(source, table)
    return np.concatenate((np.asarray(standard), np.asarray(alternate)), axis=2).transpose(2, 0, 1)


def training_thumbnails(samples, full, table, original, deadline):
    print(f"Preparing 128px whole-image augmentations for {len(full)} training images", flush=True)
    encoded = np.empty((len(full), 6, CROP, CROP), dtype=np.uint8)
    items = [samples[index] for index in full.tolist()]
    with ThreadPoolExecutor(max_workers=4) as workers:
        prepare = partial(thumbnail, table=table)
        for start in range(0, len(items), 32):
            check_time(deadline)
            for index, values in enumerate(workers.map(prepare, items[start:start + 32]), start=start):
                encoded[index] = values
    return ThumbnailTrainingViews(original, torch.from_numpy(encoded), full, len(samples))


def contrast_components(views):
    q75, alternate = (views.flatten(0, 1).float() / 255).split(3, dim=1)
    common = (q75 + alternate) * 0.5
    blur = F.avg_pool2d(common, 3, stride=1, padding=1, count_include_pad=False)
    residual = common - blur
    energy = F.avg_pool2d(residual.square(), 9, stride=1, padding=4, count_include_pad=False)
    scale = (energy + (4 / 255) ** 2).sqrt()
    return common - 0.5, residual, alternate - q75, scale


class NormalizedResponseCNN(CodecResponseCNN):
    def __init__(self, variant):
        if variant != NORMALIZED:
            raise ValueError(f"Unknown normalized response variant: {variant}")
        super().__init__(VARIANT)

    def patch_features(self, views):
        common, residual, change, scale = contrast_components(views)
        return torch.cat((common, (residual / scale).tanh(), (change / scale).tanh()), dim=1)


class HybridResponseCNN(CodecResponseCNN):
    def __init__(self, variant):
        if variant != HYBRID:
            raise ValueError(f"Unknown hybrid response variant: {variant}")
        super().__init__(VARIANT)
        self.features[0] = nn.Conv2d(15, 16, 5, stride=4, padding=2, bias=False)

    def patch_features(self, views):
        common, residual, change, scale = contrast_components(views)
        return torch.cat((common, 6 * residual, 6 * change, (residual / scale).tanh(), (change / scale).tanh()), dim=1)


def baseline(args, cohort, native, truth, device, deadline):
    rows = torch.arange(len(truth))
    if cohort in (512, 1024):
        return combined_reference(args, cohort, native, truth, None, rows, device, deadline)
    if cohort == 128:
        spatial_path = args.anchor_reference / "reference_128/spatial"
        spectrum_path = args.anchor_reference / "reference_128/spectrum"
    else:
        spatial_path = args.group_reference / "confirm_256_reference"
        spectrum_path = args.spectrum_reference / "spectrum_256"
    spatial = restore_ensemble([spatial_path / f"s{seed}/last.pt" for seed in SEEDS], device)
    spectral = restore(spectrum_path, SpectrumPatchCNN, SPEC_VARIANT, device)
    frequency = spectral_views(native, rows, deadline)
    return (scores(spatial, native, rows, device, deadline) + scores(spectral, frequency, rows, device, deadline)) / 2


def restore_candidate(folder, variant, factory, device):
    models = []
    for seed in SEEDS:
        path = folder / f"s{seed}" / "last.pt"
        state = load_checkpoint(path)
        if (state["variant"], state["seed"], state["epoch"]) != (variant, seed, SCREEN_EPOCHS):
            raise ValueError(f"Unexpected candidate checkpoint: {path}")
        model = factory(variant).to(device, memory_format=torch.channels_last)
        model.load_state_dict(state["weights"])
        models.append(model)
    return models


def run(args, device, deadline):
    # Preserve the paired real samples from earlier 512/1024/128 comparisons.
    samples, labels, sizes, anchor, folds, full, table, identity = prepare_data((512, 1024, 128, CONFIRM))
    crop_size, variant, factory = CROP_SIZE, VARIANT, CodecResponseCNN
    if args.method == "contrast-normalized":
        crop_size, variant, factory = CROP, NORMALIZED, NormalizedResponseCNN
    elif args.method == "hybrid-response":
        crop_size, variant, factory = CROP, HYBRID, HybridResponseCNN
    elif args.method == "thumbnail-augmented":
        crop_size = CROP
    trainer = partial(train_variant, model_factory=factory)
    report = {"split_sha256": identity, "crop_size": crop_size, "variant": variant, "method": args.method,
              "anchor_quantile": QUANTILE, "screen": {}}
    if args.reuse_128:
        old = read_json(args.output / "summary.json")
        old_method = old.get("method")
        if old_method is None and old["variant"] == NORMALIZED:
            old_method = "contrast-normalized"
        if (old["split_sha256"], old["variant"], old_method, old["crop_size"]) != (identity, variant, args.method, crop_size):
            raise ValueError("Existing 128px checkpoints use a different experiment configuration")

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("prepare_labeled")
    native, model_views = view_pair(samples, table, deadline, crop_size=crop_size)
    training_views = model_views
    if args.method == "thumbnail-augmented":
        training_views = training_thumbnails(samples, full, table, model_views, deadline)
    evaluated = []

    def check(cohort):
        training, val = folds[cohort]
        rows = exclude_anchor(training, labels, anchor)
        folder = args.output / f"group-{cohort}" / variant
        if cohort == 128 and args.reuse_128:
            models = restore_candidate(folder, variant, factory, device)
        else:
            models = trainer(folder, variant, training_views, labels, sizes, rows, device, deadline)
        anchor_scores = scores(models, model_views, anchor, device, deadline)
        threshold = cutoff(anchor_scores, QUANTILE)
        current = recenter(scores(models, model_views, val, device, deadline), threshold)
        old_views, _ = cache([samples[i] for i in val.tolist()], QUALITY, deadline)
        reference = baseline(args, cohort, old_views, labels[val], device, deadline)
        truth = labels[val].numpy()
        result = comparison(truth, reference, current)
        print(f"{cohort}: accuracy {result['reference']['accuracy']:.3%} -> {result['deep_fusion']['accuracy']:.3%}, "
              f"AUROC {result['reference']['auroc']:.4f} -> {result['deep_fusion']['auroc']:.4f}", flush=True)
        if cohort in SCREEN:
            evaluated.append((truth, reference, current))
        return {"threshold": threshold, "train_count": len(rows), **result}

    for cohort in SCREEN:
        record(f"screen_{cohort}")
        result = check(cohort)
        report["screen"][str(cohort)] = result
        if cohort == 128 and result["delta"]["accuracy"] < MIN_CONFIRM_GAIN:
            record("stopped_at_small_image_screen")
            return
    ranking = bootstrap_mean_gain(evaluated)
    ranking["passes"] = (ranking["mean_gain"] >= MIN_SCREEN_GAIN and ranking["bootstrap_95_percent"][0] > 0
                         and all(result["delta"]["accuracy"] > 0 for result in report["screen"].values()))
    report["gate"] = ranking
    if not ranking["passes"]:
        record("stopped_at_screen")
        return

    record("confirm_256")
    confirmation = check(CONFIRM)
    report["confirmation"] = confirmation
    if (confirmation["delta"]["accuracy"] < MIN_SCREEN_GAIN
            or confirmation["paired_accuracy"]["bootstrap_95_percent"][0] <= 0):
        record("stopped_at_confirmation")
        return

    record("full_training")
    train = exclude_anchor(full, labels, anchor)
    models = trainer(args.output / "full" / variant, variant, training_views, labels, sizes, train, device, deadline)
    threshold = cutoff(scores(models, model_views, anchor, device, deadline), QUANTILE)
    report["train_count"] = len(train)
    report["calibrated_threshold"] = threshold
    del model_views, native, training_views
    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    _, paired = view_pair(test, table, deadline, crop_size=crop_size)
    values = scores(models, paired, torch.arange(len(test)), device, deadline)
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(score > threshold)) for sample, score in zip(test, values, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--method", choices=("smaller-crops", "contrast-normalized", "thumbnail-augmented", "hybrid-response"),
                        default="smaller-crops")
    parser.add_argument("--reuse-128", action="store_true", help="Reuse the completed 128px checkpoints from this run")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    parser.add_argument("--anchor-reference", type=Path, default=Path("lqy/runs/codec-anchor/round-01"))
    parser.add_argument("--group-reference", type=Path, default=Path("lqy/runs/group-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

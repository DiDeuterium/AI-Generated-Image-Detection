"""Train and verify a scratch detector under class-matched imaging conditions."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch import nn
from torchvision.models import mobilenet_v2

from detection.data import jpeg_roundtrip
from detection.training import write_json

from .canonical_augmentation import CORRELATION_VARIANT, CorrelationCNN, calibration_indices
from .cnn import device_for, read_json, run_lock
from .codec_anchor import prepare_data, recenter
from .codec_canonical import balanced_cutoff
from .codec_patch import encode
from .codec_response import PairedViews
from .codec_scale import restore_candidate
from .deep_fusion import comparison
from .jpeg_history import jpeg_profile
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, check_time, fit, image_patches, scores
from .source_holdout import SourceCNN, source_folds, source_gain
from .spectrum_patch import QUALITY


RUNS = Path("lqy/runs")
MATCHED_TRAIN = ((173, "jpeg", 72, 2, 0.025), (221, "jpeg", 87, 0, 0.065),
                 (283, "webp", 79, 0, 0.045), (241, "jpeg", 92, 1, 0.015))
CALIBRATION = (211, "jpeg", 78, 2, 0.048)
MATCHED_VALIDATION = {
    "jpeg81": (187, "jpeg", 81, 1, 0.033),
    "webp84": (257, "webp", 84, 0, 0.057),
}
WINDOW_SIDE = 112
WINDOW_POSITIONS = ((0.5, 0.5), (0.2, 0.2), (0.8, 0.25), (0.25, 0.8), (0.8, 0.8))
SPATIAL_VARIANT = "matched_spatial_correlation"
SCENE_VARIANT = "matched_scene_context"


class SpatialCorrelationCNN(SourceCNN):
    """Measure spatial feature correlations at several fixed offsets."""

    def __init__(self, variant):
        if variant != SPATIAL_VARIANT:
            raise ValueError("Unexpected spatial correlation architecture")
        super().__init__("constrained_rgb")
        channels = 16
        self.projection = nn.Conv2d(64, channels, 1, bias=False)
        dimensions = 128 + 6 * channels
        self.classifier = nn.Sequential(nn.LayerNorm(dimensions), nn.Linear(dimensions, 64), nn.GELU(),
                                        nn.Dropout(0.15), nn.Linear(64, 2))

    def patch_embeddings(self, views):
        pixels = self.patch_features(views)
        mapped = self.features(pixels.contiguous(memory_format=torch.channels_last))
        residual = self.projection(mapped).float()
        residual = residual - residual.mean(dim=(-2, -1), keepdim=True)
        residual = residual * (residual.square().mean(dim=(-2, -1), keepdim=True) + 0.01).rsqrt()
        measurements = []
        for offset in (1, 2, 3):
            horizontal = residual[..., offset:] * residual[..., :-offset]
            vertical = residual[..., offset:, :] * residual[..., :-offset, :]
            measurements.extend((horizontal.mean(dim=(-2, -1)), vertical.mean(dim=(-2, -1))))
        flat = mapped.float().flatten(2)
        return torch.cat((flat.mean(dim=2), flat.std(dim=2, correction=0), *measurements), dim=1)


class SceneContextCNN(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant != SCENE_VARIANT:
            raise ValueError("Unexpected scene analysis architecture")
        self.features = mobilenet_v2(weights=None, width_mult=0.75, num_classes=2)

    def forward(self, views):
        rgb = (views[:, 0, :3].float() - 127.5) / 127.5
        return self.features(rgb.contiguous(memory_format=torch.channels_last))


def matched_image(rgb, profile):
    side, codec, quality, subsampling, trim = profile
    width, height = rgb.size
    extent = min(width, height) * (1 - 2 * trim)
    left = (width - extent) * 0.43
    top = (height - extent) * 0.57
    # Shift the original pixel grid before equally resizing and encoding both classes.
    rgb = rgb.filter(ImageFilter.GaussianBlur(radius=0.40))
    rgb = rgb.transform((side, side), Image.Transform.EXTENT, (left, top, left + extent, top + extent),
                        resample=Image.Resampling.BICUBIC)
    if codec == "jpeg":
        return jpeg_roundtrip(rgb, quality=quality, subsampling=subsampling)
    with io.BytesIO() as stream:
        rgb.save(stream, format="WEBP", method=1, quality=quality)
        stream.seek(0)
        with Image.open(stream) as decoded:
            return decoded.convert("RGB")


def initial_jpeg(rgb, profile):
    sampling, luminance, chrominance = profile
    with io.BytesIO() as stream:
        rgb.save(stream, format="JPEG", quality=-1, subsampling=sampling,
                 qtables=[list(luminance), list(chrominance)])
        stream.seek(0)
        with Image.open(stream) as decoded:
            return decoded.convert("RGB")


def window_pixels(rgb, profile, crops, encoder):
    primary, alternate = [], []
    width, height = rgb.size
    if min(width, height) < WINDOW_SIDE:
        raise ValueError("A native-window comparison requires a full window in each image")
    for index in crops:
        x, y = WINDOW_POSITIONS[index]
        left = round((width - WINDOW_SIDE) * x)
        top = round((height - WINDOW_SIDE) * y)
        region = rgb.crop((left, top, left + WINDOW_SIDE, top + WINDOW_SIDE))
        normalized = matched_image(region, profile)
        center = (normalized.width - CROP) // 2
        crop = normalized.crop((center, center, center + CROP, center + CROP))
        primary.append(np.asarray(jpeg_roundtrip(crop, quality=QUALITY, subsampling=2)).transpose(2, 0, 1))
        alternate.append(np.asarray(encoder(crop)).transpose(2, 0, 1))
    return np.stack(primary), np.stack(alternate)


def cache_views(items, profiles, table, deadline, *, crops, native_window=False,
                initial_profiles=None, alternate_profiles=None, scene_context=False, input_transform=None):
    settings = (profiles,) if alternate_profiles is None else (profiles, alternate_profiles)
    one = np.empty((len(settings), len(items), len(crops), 3, CROP, CROP), dtype=np.uint8)
    two = np.empty_like(one)
    encoder = partial(encode, table=table)

    def render_profile(rgb, profile):
        if native_window:
            return window_pixels(rgb, profile, crops, encoder)
        processed = matched_image(rgb, profile)
        if scene_context:
            blurred = processed.filter(ImageFilter.GaussianBlur(radius=1.2))
            canvas = blurred.resize((CROP, CROP), Image.Resampling.BICUBIC)
            values = np.asarray(canvas).transpose(2, 0, 1)[None]
            return values, values
        return image_patches(processed, QUALITY)[crops], image_patches(processed, QUALITY, encoder=encoder)[crops]

    def render(item, views, initial):
        with Image.open(item.path) as opened:
            rgb = opened.convert("RGB")
        if input_transform is not None:
            rgb = input_transform(rgb)
        if initial is not None:
            rgb = initial_jpeg(rgb, initial)
        return [render_profile(rgb, profile) for profile in views]

    history = [None] * len(items) if initial_profiles is None else initial_profiles
    if len(history) != len(items) or any(len(group) != len(items) for group in settings):
        raise ValueError("Every image needs one acquisition profile")
    settings_by_image = list(zip(*settings, strict=True))
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(items), 32):
            check_time(deadline)
            stop = start + 32
            batch = workers.map(render, items[start:stop], settings_by_image[start:stop], history[start:stop])
            for row, results in enumerate(batch, start=start):
                for stage, (primary, alternate) in enumerate(results):
                    one[stage, row], two[stage, row] = primary, alternate
    views = tuple(PairedViews(torch.from_numpy(a), torch.from_numpy(b)) for a, b in zip(one, two, strict=True))
    return views[0] if alternate_profiles is None else views


def training_profiles(labels):
    halfway = len(labels) // 2
    if not halfway or len(labels) % 2 or (labels[:halfway] != 0).any() or (labels[halfway:] != 1).any():
        raise ValueError("Expected equal, separately ordered classes for matched transforms")
    original = [MATCHED_TRAIN[i % len(MATCHED_TRAIN)] for i in range(halfway)]
    alternate = [MATCHED_TRAIN[(i + 1) % len(MATCHED_TRAIN)] for i in range(halfway)]
    return original * 2, alternate * 2


def non_small_pairs(rows, sizes):
    photo, generated = rows.chunk(2)
    eligible = torch.tensor([min(*sizes[a], *sizes[b]) >= WINDOW_SIDE
                             for a, b in zip(photo.tolist(), generated.tolist(), strict=True)], dtype=torch.bool)
    return torch.cat((photo[eligible], generated[eligible]))


def training_histories(items):
    halfway = len(items) // 2
    with ThreadPoolExecutor(max_workers=4) as workers:
        profiles = list(workers.map(jpeg_profile, items[:halfway]))
    fallback = next((profile for profile in profiles if profile is not None), None)
    if fallback is None:
        raise ValueError("No photographic JPEG profile is available")
    return [None] * halfway + [profile or fallback for profile in profiles]


def calibrated(models, packed, labels, device, deadline):
    count = len(labels)
    probability = scores(models, packed, torch.arange(count), device, deadline)
    return balanced_cutoff(probability, labels.numpy())


def run(args, device, deadline):
    samples, labels, sizes, _, _, full, table, identity = prepare_data()
    meta = read_json(args.control / "summary.json")
    if meta["split_sha256"] != identity or meta["stage"] != "complete" or meta["variant"] != CORRELATION_VARIANT:
        raise ValueError("Matched control has an unexpected training split or architecture")
    variant, factory = ((SPATIAL_VARIANT, SpatialCorrelationCNN) if args.spatial_correlation
                        else (CORRELATION_VARIANT, CorrelationCNN))
    if args.scene_context:
        variant, factory = SCENE_VARIANT, SceneContextCNN
    report = {"split_sha256": identity, "training_profiles": MATCHED_TRAIN,
              "validation_profiles": MATCHED_VALIDATION, "variant": variant,
              "native_window_side": WINDOW_SIDE if args.native_window else None,
              "matched_original_jpeg_history": args.match_history, "cohorts": {}}
    cache = partial(cache_views, native_window=args.native_window)
    candidate_cache = partial(cache, scene_context=args.scene_context)
    train_crops = [0] if args.scene_context else [0, 1, 4]
    validation_crops = [0] if args.scene_context else [0, 1, 2, 3, 4]

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    independent = []
    for name, (training, photos, validation) in source_folds(samples, labels, sizes, len(full)).items():
        record(f"training_{name}")
        anchors = calibration_indices(samples, labels, sizes, len(full), name, training, photos)
        if args.native_window:
            training, anchors, validation = (non_small_pairs(rows, sizes) for rows in (training, anchors, validation))
        items = [samples[i] for i in training.tolist()]
        print(f"{name}: rendering {len(items)} labeled, class-matched training images", flush=True)
        profile_a, profile_b = training_profiles(labels[training])
        history = training_histories(items) if args.match_history else None
        original, alternate = candidate_cache(items, profile_a, table, deadline, crops=train_crops,
                                              initial_profiles=history, alternate_profiles=profile_b)
        indexes = torch.arange(len(items))
        candidate = [fit(args.output / name / f"s{seed}", variant, seed, SCREEN_EPOCHS,
                         original, labels[training], indexes, device, deadline,
                         model_factory=factory, paired_data=alternate) for seed in SEEDS]
        del original, alternate

        reference = restore_candidate(args.control / name, CORRELATION_VARIANT, CorrelationCNN, device)
        items = [samples[i] for i in anchors.tolist()]
        anchor_views = cache(items, [CALIBRATION] * len(items), table, deadline, crops=[0, 1, 2, 3, 4])
        control_cutoff = calibrated(reference, anchor_views, labels[anchors], device, deadline)
        if args.scene_context:
            anchor_views = candidate_cache(items, [CALIBRATION] * len(items), table, deadline, crops=validation_crops)
        candidate_cutoff = calibrated(candidate, anchor_views, labels[anchors], device, deadline)
        del anchor_views

        items = [samples[i] for i in validation.tolist()]
        indexes = torch.arange(len(items))
        truth = labels[validation].numpy()
        results, pairs = {}, []
        for profile, settings in MATCHED_VALIDATION.items():
            packed = cache(items, [settings] * len(items), table, deadline, crops=[0, 1, 2, 3, 4])
            before = recenter(scores(reference, packed, indexes, device, deadline), control_cutoff)
            if args.scene_context:
                packed = candidate_cache(items, [settings] * len(items), table, deadline, crops=validation_crops)
            after = recenter(scores(candidate, packed, indexes, device, deadline), candidate_cutoff)
            pairs.append((before, after))
            measured = comparison(truth, before, after)
            results[profile] = measured
            print(f"{name} {profile}: {measured['reference']['accuracy']:.2%} -> "
                  f"{measured['deep_fusion']['accuracy']:.2%}", flush=True)
        independent.append((truth, pairs))
        report["cohorts"][name] = {"profiles": results, "validation_count": len(validation),
                                  "training_count": len(training), "candidate_threshold": candidate_cutoff}
        record(f"verified_{name}")
    report["source_gain"] = source_gain(independent)
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--native-window", action="store_true")
    parser.add_argument("--match-history", action="store_true")
    parser.add_argument("--spatial-correlation", action="store_true")
    parser.add_argument("--scene-context", action="store_true")
    parser.add_argument("--control", type=Path, default=RUNS / "canonical-augmentation/correlation-01")
    args = parser.parse_args()
    if args.match_history and not args.native_window:
        parser.error("--match-history expects --native-window")
    if args.spatial_correlation and not args.native_window:
        parser.error("--spatial-correlation expects --native-window")
    if args.scene_context and args.native_window:
        parser.error("--scene-context expects a full normalized frame")
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

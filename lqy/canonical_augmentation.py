"""Test canonicalized training across independent labeled source groups."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
import io
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

from detection.data import jpeg_roundtrip
from detection.training import write_json

from .cnn import device_for, read_json, run_lock
from .codec_anchor import prepare_data, recenter
from . import codec_canonical
from .codec_canonical import CANONICAL, balanced_cutoff, transform
from .codec_patch import encode
from .codec_response import PairedViews
from .codec_scale import restore_candidate
from .codec_support import tagged
from .deep_fusion import comparison
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, check_time, fit, image_patches, scores
from .source_holdout import SourceCNN, source_folds, source_gain
from .spectrum_patch import QUALITY, VARIANT as SPECTRUM_VARIANT, SpectrumPatchCNN, spectrum_channels


VARIANT = "constrained_rgb"
JOINT_VARIANT = "joint_geometry_rgb"
CORRELATION_VARIANT = "correlation_rgb"
SPECTRAL_VARIANT = "joint_spectral_rgb"
TRAIN_AUGMENTATIONS = ("jpeg68", "down_rgb", "up_webp", "blur_jpeg")
VALIDATION_PROFILES = ("native", "resize_jpeg", "webp")
TRAIN_CROPS = (0, 1, 4)
GEOMETRY_SIDE = 192


def training_alteration(image, profile):
    if profile in ("down_rgb", "up_webp"):
        factor = 0.72 if profile == "down_rgb" else 1.18
        size = tuple(max(1, round(n * factor)) for n in image.size)
        image = image.resize(size, Image.Resampling.LANCZOS)
    if profile == "up_webp":
        with io.BytesIO() as stream:
            image.save(stream, format="WEBP", quality=74, method=1)
            stream.seek(0)
            with Image.open(stream) as decoded:
                return decoded.convert("RGB")
    if profile == "blur_jpeg":
        image = image.filter(ImageFilter.GaussianBlur(radius=0.65))
        return jpeg_roundtrip(image, quality=87, subsampling=0)
    if profile == "jpeg68":
        return jpeg_roundtrip(image, quality=68, subsampling=1)
    return image


def normalized_geometry(image):
    return ImageOps.fit(image, (GEOMETRY_SIDE, GEOMETRY_SIDE), method=Image.Resampling.LANCZOS)


def acquisition_alteration(image, assignment):
    factor = 0.58 + 0.86 * assignment[0] / 255
    size = tuple(max(1, round(n * factor)) for n in image.size)
    kernels = (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS)
    image = image.resize(size, kernels[assignment[1] % len(kernels)])
    if assignment[2] % 4 == 0:
        return image
    return jpeg_roundtrip(image, quality=69 + assignment[3] % 27, subsampling=(0, 1, 2)[assignment[4] % 3])


def encode_resampling_response(crop):
    coarse = crop.resize(tuple(round(size * 0.75) for size in crop.size), Image.Resampling.LANCZOS)
    recovered = coarse.resize(crop.size, Image.Resampling.BICUBIC)
    return jpeg_roundtrip(recovered, quality=QUALITY, subsampling=2)


def canonical_views(items, table, deadline, profile="native", *, paired=False, standardize_geometry=False,
                    wide_acquisition=False, alter_both=False, resample_response=False,
                    acquisition=acquisition_alteration):
    crops = len(TRAIN_CROPS) if paired else 5
    arrays = [np.empty((len(items), crops, 3, CROP, CROP), dtype=np.uint8) for _ in range(4 if paired else 2)]
    encoder = encode_resampling_response if resample_response else partial(encode, table=table)
    crop_positions = list(TRAIN_CROPS)

    def patches(rgb):
        if standardize_geometry:
            rgb = normalized_geometry(rgb)
        canon = jpeg_roundtrip(rgb, quality=CANONICAL, subsampling=2)
        first = image_patches(canon, QUALITY)
        second = image_patches(canon, QUALITY, encoder=encoder)
        return (first[crop_positions], second[crop_positions]) if paired else (first, second)

    def render(item):
        with Image.open(item.path) as opened:
            rgb = opened.convert("RGB")
        if not paired:
            return patches(transform(rgb, profile))
        assignment = sha256(f"canonical-augmentation:{item.image_id}".encode()).digest()
        if standardize_geometry or wide_acquisition or alter_both:
            changed = acquisition(rgb, assignment)
        else:
            changed = training_alteration(rgb, TRAIN_AUGMENTATIONS[assignment[0] % len(TRAIN_AUGMENTATIONS)])
        if alter_both:
            rgb = acquisition(rgb, assignment[5:])
        return (*patches(rgb), *patches(changed))

    print(f"Preparing {profile} canonical pairs for {len(items)} labeled images", flush=True)
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(items), 32):
            check_time(deadline)
            for position, parts in enumerate(workers.map(render, items[start:start + 32]), start=start):
                for array, pixels in zip(arrays, parts, strict=True):
                    array[position] = pixels

    tensors = tuple(map(torch.from_numpy, arrays))
    original = PairedViews(*tensors[:2])
    return (original, PairedViews(*tensors[2:])) if paired else original


class JointViews:
    def __init__(self, native, normalized):
        self.native = native
        self.normalized = normalized

    def __getitem__(self, rows):
        return torch.cat((self.native[rows], self.normalized[rows]), dim=-3)


def joint_views(items, table, deadline, profile="native", *, paired=False):
    raw = canonical_views(items, table, deadline, profile, paired=paired, wide_acquisition=True)
    normalized = canonical_views(items, table, deadline, profile, paired=paired, standardize_geometry=True)
    if paired:
        return tuple(JointViews(a, b) for a, b in zip(raw, normalized, strict=True))
    return JointViews(raw, normalized)


def add_normalized_views(native, items, profile, table, deadline):
    return JointViews(native, canonical_views(items, table, deadline, profile, standardize_geometry=True))


def use_resampling_response(_canonical, items, profile, table, deadline):
    return canonical_views(items, table, deadline, profile, resample_response=True)


def spectral_pair(native, deadline):
    first = native.reference
    spectra = np.empty(first.shape, dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=3) as workers:
        for start in range(0, len(first), 16):
            check_time(deadline)
            patches = first[start:start + 16].numpy()
            for index, transformed in enumerate(workers.map(spectrum_channels, patches), start=start):
                spectra[index] = transformed
    return JointViews(native, torch.from_numpy(spectra))


def spectral_response_views(items, table, deadline, profile="native", *, paired=False):
    views = canonical_views(items, table, deadline, profile, paired=paired, wide_acquisition=True)
    print(f"Preparing normalized spectra for {len(items)} images", flush=True)
    if paired:
        return tuple(spectral_pair(view, deadline) for view in views)
    return spectral_pair(views, deadline)


def add_spectral_views(native, _items, _profile, _table, deadline):
    return spectral_pair(native, deadline)


class JointGeometryCNN(nn.Module):
    def __init__(self, variant, *, second=None):
        super().__init__()
        if variant != JOINT_VARIANT:
            raise ValueError(f"Unknown joint view model: {variant}")
        self.native = SourceCNN(VARIANT)
        self.normalized = SourceCNN(VARIANT) if second is None else second
        self.fusion = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 48), nn.GELU(),
                                    nn.Dropout(0.15), nn.Linear(48, 2))

    def split_views(self, inputs):
        return inputs.split(6, dim=2)

    def branches(self, inputs):
        raw, normalized = self.split_views(inputs)
        raw_features = self.native.patch_embeddings(raw).reshape(len(raw), raw.shape[1], -1).mean(dim=1)
        geo_features = self.normalized.patch_embeddings(normalized).reshape(len(raw), raw.shape[1], -1).mean(dim=1)
        return (self.fusion(torch.cat((raw_features, geo_features), dim=1)),
                self.native.classifier(raw_features), self.normalized.classifier(geo_features))

    def forward(self, inputs):
        combined, native, normalized = self.branches(inputs)
        return 0.6 * combined + 0.25 * native + 0.15 * normalized


class JointSpectralCNN(JointGeometryCNN):
    def __init__(self, variant):
        if variant != SPECTRAL_VARIANT:
            raise ValueError(f"Unknown joint spectral model: {variant}")
        super().__init__(JOINT_VARIANT, second=SpectrumPatchCNN(SPECTRUM_VARIANT))

    def split_views(self, inputs):
        primary = inputs[:, :, :6]
        spectrum = torch.cat((inputs[:, :, :3], inputs[:, :, 6:]), dim=2)
        return primary, spectrum


class CorrelationCNN(SourceCNN):
    def __init__(self, variant):
        if variant != CORRELATION_VARIANT:
            raise ValueError(f"Unknown correlation model: {variant}")
        super().__init__(VARIANT)
        channels = 16
        self.projection = nn.Conv2d(64, channels, 1, bias=False)
        self.register_buffer("pairs", torch.triu_indices(channels, channels, offset=1))
        pooled_size = 128 + self.pairs.shape[1]
        self.classifier = nn.Sequential(nn.LayerNorm(pooled_size), nn.Linear(pooled_size, 64), nn.GELU(),
                                        nn.Dropout(0.15), nn.Linear(64, 2))

    def patch_embeddings(self, views):
        pixels = self.patch_features(views)
        mapped = self.features(pixels.contiguous(memory_format=torch.channels_last))
        coefficients = self.projection(mapped).float().flatten(2)
        centered = coefficients - coefficients.mean(dim=2, keepdim=True)
        scaled = centered * (centered.square().mean(dim=2, keepdim=True) + 0.01).rsqrt()
        covariance = (scaled @ scaled.transpose(1, 2)) / scaled.shape[2]
        pooled = mapped.float().flatten(2)
        return torch.cat((pooled.mean(dim=2), pooled.std(dim=2, correction=0),
                          covariance[:, self.pairs[0], self.pairs[1]]), dim=1)


def joint_loss(model, original, alternate, labels, _ids):
    target = labels.repeat(2)
    combined, raw, geo = model.branches(torch.cat((original, alternate)))
    logits = 0.6 * combined + 0.25 * raw + 0.15 * geo
    first, second = logits.float().softmax(dim=1).chunk(2)
    branches = F.cross_entropy(raw, target) + F.cross_entropy(geo, target)
    return F.cross_entropy(logits, target) + 0.2 * branches + 0.5 * (first - second).square().mean()


def worst_profile_loss(model, original, altered, labels, _ids):
    logits = model(torch.cat((original, altered))).float()
    target = labels.repeat(2)
    losses = F.cross_entropy(logits, target, reduction="none")
    groups = F.one_hot(torch.cat((labels, labels + 2)), num_classes=4).to(losses.dtype)
    counts = groups.sum(dim=0)
    risks = (groups.T @ losses) / counts.clamp(min=1)
    temperature = 0.20
    robust = temperature * ((risks / temperature).masked_fill(counts == 0, -torch.inf).logsumexp(dim=0)
                            - (counts > 0).sum().float().log())
    first, second = logits.softmax(dim=1).chunk(2)
    return 0.55 * losses.mean() + 0.45 * robust + 0.5 * (first - second).square().mean()


def calibration_indices(samples, labels, sizes, end, name, training, photographs):
    trained = set(training.tolist())

    def excluded(index):
        if name == "native_1024":
            return sizes[index] == (1024, 1024)
        return sizes[index] == (256, 256) and tagged(samples[index])

    generated = (i for i in range(end) if labels[i] == 1 and i not in trained and not excluded(i))
    ordered = sorted(generated, key=lambda i: sha256(f"both-class-canonical:{samples[i].image_id}".encode()).digest())
    if len(ordered) < len(photographs):
        raise ValueError(f"Missing independent generated calibration examples for {name}")
    return torch.cat((photographs, torch.tensor(ordered[:len(photographs)])))


def run(args, device, deadline):
    samples, labels, sizes, _, _, full, table, identity = prepare_data()
    control = read_json(args.control / "summary.json")
    calibration = read_json(args.calibration_reference)
    if any(record["split_sha256"] != identity for record in (control, calibration)) or control["stage"] != "complete":
        raise ValueError("Canonical training and the matched controls require the same labeled split")
    variant, factory, objective = VARIANT, SourceCNN, None
    alter_both = args.alter_both or args.resample_response
    candidate_views = partial(canonical_views, standardize_geometry=args.standardize_geometry,
                              alter_both=alter_both, resample_response=args.resample_response)
    if args.joint_geometry:
        variant, factory, objective, candidate_views = JOINT_VARIANT, JointGeometryCNN, joint_loss, joint_views
    if args.correlation:
        variant, factory = CORRELATION_VARIANT, CorrelationCNN
        candidate_views = partial(canonical_views, wide_acquisition=True)
    if args.joint_spectrum:
        variant, factory, objective = SPECTRAL_VARIANT, JointSpectralCNN, joint_loss
        candidate_views = spectral_response_views
    if args.worst_profile:
        objective = worst_profile_loss
        candidate_views = partial(canonical_views, wide_acquisition=True)
    report = {"split_sha256": identity, "variant": variant, "augmentations": TRAIN_AUGMENTATIONS, "cohorts": {}}
    if args.standardize_geometry or args.joint_geometry:
        report["geometry_side"] = GEOMETRY_SIDE
        report["augmentations"] = {"resize_factor": [0.58, 1.44], "jpeg_quality": [69, 95]}
    if alter_both or args.correlation or args.joint_spectrum or args.worst_profile:
        report["augmentations"] = {"resize_factor": [0.58, 1.44], "jpeg_quality": [69, 95]}
    if args.worst_profile:
        report["worst_profile_loss"] = True
    if alter_both:
        report["alter_both"] = True
    if args.resample_response:
        report["resample_response"] = True

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    transfers = []
    for name, (training, photos, validation) in source_folds(samples, labels, sizes, len(full)).items():
        record(f"training_{name}")
        selected = [samples[i] for i in training.tolist()]
        base, other = candidate_views(selected, table, deadline, paired=True)
        positions = torch.arange(len(training))
        models = [fit(args.output / name / f"s{seed}", variant, seed, SCREEN_EPOCHS, base, labels[training],
                      positions, device, deadline, model_factory=factory, paired_data=other,
                      paired_objective=objective) for seed in SEEDS]
        del base, other
        anchors = calibration_indices(samples, labels, sizes, len(full), name, training, photos)
        data = candidate_views([samples[i] for i in anchors.tolist()], table, deadline)
        target_scores = scores(models, data, torch.arange(len(anchors)), device, deadline)
        threshold = balanced_cutoff(target_scores, labels[anchors].numpy())
        del data

        previous = restore_candidate(args.control / name / VARIANT, VARIANT, SourceCNN, device)
        previous_threshold = calibration["cohorts"][name]["models"][VARIANT]["rules"]["both_class"]["threshold"]
        items = [samples[i] for i in validation.tolist()]
        truth = labels[validation].numpy()
        positions = torch.arange(len(items))
        pairs, results = [], {}
        for profile in VALIDATION_PROFILES:
            data = canonical_views(items, table, deadline, profile)
            before = recenter(scores(previous, data, positions, device, deadline), previous_threshold)
            if args.standardize_geometry or args.joint_geometry or args.resample_response or args.joint_spectrum:
                data = candidate_views(items, table, deadline, profile)
            after = recenter(scores(models, data, positions, device, deadline), threshold)
            pairs.append((before, after))
            results[profile] = comparison(truth, before, after)
            print(f"{name} {profile}: {results[profile]['reference']['accuracy']:.2%} -> "
                  f"{results[profile]['deep_fusion']['accuracy']:.2%}", flush=True)
        transfers.append((truth, pairs))
        report["cohorts"][name] = {"threshold": threshold, "count": len(validation), "profiles": results}
        record(f"validated_{name}")
    report["aggregate"] = source_gain(transfers)
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--control", type=Path, default=Path("lqy/runs/source-holdout/round-01"))
    parser.add_argument("--calibration-reference", type=Path,
                        default=Path("lqy/runs/source-holdout/canonical-robust-calibration.json"))
    parser.add_argument("--full-output", type=Path, help="Verify full training if the paired augmentation improves source transfer")
    parser.add_argument("--promote", action="store_true", help="Verify an existing source pilot without retraining it")
    parser.add_argument("--test-source-tradeoff", action="store_true",
                        help="Check a correlation candidate with stable native and WebP gains on full labeled data")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--standardize-geometry", action="store_true")
    mode.add_argument("--joint-geometry", action="store_true")
    mode.add_argument("--alter-both", action="store_true")
    mode.add_argument("--resample-response", action="store_true")
    mode.add_argument("--correlation", action="store_true")
    mode.add_argument("--joint-spectrum", action="store_true")
    mode.add_argument("--worst-profile", action="store_true")
    args = parser.parse_args()
    if args.promote and args.full_output is None:
        parser.error("--promote requires --full-output")
    if args.test_source_tradeoff and not args.promote:
        parser.error("--test-source-tradeoff requires --promote")
    device = device_for(args.device, 6)
    if not args.promote:
        with run_lock(args.output):
            run(args, device, time.monotonic() + 3600)
    if args.full_output is not None:
        verify_full(args.output, args.full_output, args.calibration_reference, device,
                    test_tradeoff=args.test_source_tradeoff)


def verify_full(pilot_dir, output, calibration_reference, device, *, test_tradeoff=False):
    pilot = read_json(pilot_dir / "summary.json")
    if pilot["stage"] != "complete":
        raise ValueError("The source pilot must finish before full training")
    gain = pilot["aggregate"]
    cohorts = pilot["cohorts"]
    resize_gains = [group["profiles"]["resize_jpeg"]["delta"]["accuracy"] for group in cohorts.values()]
    native_gains = [group["profiles"]["native"]["delta"]["accuracy"] for group in cohorts.values()]
    geometry = "geometry_side" in pilot
    joint = pilot["variant"] == JOINT_VARIANT
    correlation = pilot["variant"] == CORRELATION_VARIANT
    spectrum = pilot["variant"] == SPECTRAL_VARIANT
    profile_loss = pilot.get("worst_profile_loss", False)
    alter_both = pilot.get("alter_both", False)
    response = pilot.get("resample_response", False)
    if test_tradeoff:
        webp_gains = [group["profiles"]["webp"]["delta"]["accuracy"] for group in cohorts.values()]
        if not correlation or min(native_gains + webp_gains) < 0.05:
            return
    elif geometry or alter_both or correlation or profile_loss:
        if min(resize_gains) < 0.025 or np.mean(resize_gains) < 0.08:
            return
    elif (gain["mean_gain"] < 0.025 or gain["paired_95_percent"][0] <= 0
          or min(resize_gains) <= 0 or min(native_gains) < -0.03):
        return
    training_loader = partial(canonical_views, paired=True, standardize_geometry=geometry,
                              alter_both=alter_both, resample_response=response,
                              wide_acquisition=correlation or profile_loss)
    calibration_loader = partial(canonical_views, standardize_geometry=geometry,
                                 resample_response=response) if geometry or response else None
    candidate_adapter = use_resampling_response if response else None
    if joint:
        training_loader, calibration_loader = partial(joint_views, paired=True), joint_views
        candidate_adapter = add_normalized_views
    if spectrum:
        training_loader = partial(spectral_response_views, paired=True)
        calibration_loader, candidate_adapter = spectral_response_views, add_spectral_views
    objective = joint_loss if joint or spectrum else (worst_profile_loss if profile_loss else None)
    factory = {JOINT_VARIANT: JointGeometryCNN, CORRELATION_VARIANT: CorrelationCNN,
               SPECTRAL_VARIANT: JointSpectralCNN}.get(pilot["variant"])
    output_args = argparse.Namespace(
        output=output, calibration_reference=calibration_reference,
        narrow_reference=Path("lqy/runs/codec-final/upstream-trained"),
        near_reference=Path("lqy/runs/codec-final/near-upstream"),
        night_reference=Path("lqy/runs/night-train/2026-11-21"),
        spectrum_reference=Path("lqy/runs/spectrum-patch/round-01"),
    )
    with run_lock(output):
        codec_canonical.run(output_args, device, time.monotonic() + 3600,
                            training_loader=training_loader,
                            augmentation_check={"gain": gain, "resize_gains": resize_gains,
                                                "geometry_side": pilot.get("geometry_side"),
                                                "alter_both": alter_both, "resample_response": response,
                                                "worst_profile_loss": profile_loss,
                                                "test_source_tradeoff": test_tradeoff},
                            candidate_transform=normalized_geometry if geometry and not joint else None,
                            calibration_loader=calibration_loader,
                            model_variant=pilot["variant"],
                            model_factory=factory,
                            paired_objective=objective,
                            candidate_adapter=candidate_adapter,
                            screen_weights=(0.25, *codec_canonical.WEIGHTS) if test_tradeoff else codec_canonical.WEIGHTS)


if __name__ == "__main__":
    main()

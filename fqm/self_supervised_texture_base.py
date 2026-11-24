"""Jointly classify images and predict pre-degradation patch texture from training images."""

import argparse
from functools import partial
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detection.training import seed_all
from lqy.canonical_augmentation import TRAIN_CROPS, canonical_views
from lqy.codec_patch import labeled_data, tables_by_frequency
from lqy.night_train import CROP
from lqy_resize_retrain import training_split
from native_texture_base import NativeTextureBase, SEED, acquisition, batch_view, consistency_loss, fit


MODEL = "self_supervised_texture_base"
AUTOCORRELATION_MODEL = "self_supervised_autocorrelation_texture"
DIRECTIONAL_MODEL = "self_supervised_directional_texture"
PHASE_STEM_MODEL = "self_supervised_phase_stem_texture"
AUXILIARY_WEIGHT = .1
TARGET_COUNT = 2
DIRECTIONAL_TARGET_COUNT = 4


class SelfSupervisedTexture(NativeTextureBase):
    def __init__(self, *, target_count=TARGET_COUNT):
        super().__init__()
        width = self.attention.in_features
        self.texture_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, target_count))

    def texture_embeddings(self, views):
        return self.patch_embeddings(views).reshape(len(views), views.shape[1], -1)

    def classify_and_predict_texture(self, views):
        patches = self.texture_embeddings(views)
        return self.classifier(self.pool_embeddings(patches)), self.texture_head(patches)


class PhaseStemTexture(SelfSupervisedTexture):
    def __init__(self):
        super().__init__()
        conv, norm, activation = self.features[0]
        channels = conv.out_channels
        conv.stride = (1, 1)
        mix = nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False)
        with torch.no_grad():
            # Begin with the phase sampled by the original stride-two convolution.
            mix.weight.zero_()
            mix.weight[:, ::4, 0, 0].copy_(torch.eye(channels))
        self.features[0] = nn.Sequential(conv, nn.PixelUnshuffle(2), mix, norm, activation)


def patch_boxes(size):
    width, height = size
    right, bottom = max(0, width - CROP), max(0, height - CROP)
    pad_x, pad_y = max(0, CROP - width) // 2, max(0, CROP - height) // 2
    positions = ((right // 2, bottom // 2), (0, 0), (right, 0), (0, bottom), (right, bottom))
    for index in TRAIN_CROPS:
        x, y = positions[index]
        yield max(0, x - pad_x), max(0, y - pad_y), min(width, x + CROP - pad_x), min(height, y + CROP - pad_y)


def texture_energy(luma):
    height, width = luma.shape
    if min(height, width) < 4:
        luma = np.pad(luma, ((0, max(0, 4 - height)), (0, max(0, 4 - width))), mode="edge")
        height, width = luma.shape
    coarse = luma[:height // 2 * 2, :width // 2 * 2].reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))

    def log_rms(pixels):
        horizontal = np.square(np.diff(pixels, axis=1)).mean()
        vertical = np.square(np.diff(pixels, axis=0)).mean()
        return np.log1p(np.sqrt(.5 * (horizontal + vertical)))

    return log_rms(luma), log_rms(coarse)


def directional_residual_autocorrelation(luma):
    height, width = luma.shape
    if min(height, width) < 5:
        luma = np.pad(luma, ((0, max(0, 5 - height)), (0, max(0, 5 - width))), mode="edge")
    center = luma[1:-1, 1:-1]
    neighbors = .25 * (luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:])
    residual = center - neighbors
    residual -= residual.mean()

    def correlation(first, second):
        numerator = (first * second).mean()
        energy = np.sqrt(np.square(first).mean() * np.square(second).mean())
        return numerator / max(float(energy), 1.0)

    return tuple(value for lag in (1, 2) for value in (
        correlation(residual[:, :-lag], residual[:, lag:]), correlation(residual[:-lag], residual[lag:]),
    ))


def residual_autocorrelation(luma):
    horizontal_near, vertical_near, horizontal_far, vertical_far = directional_residual_autocorrelation(luma)
    return .5 * (horizontal_near + vertical_near), .5 * (horizontal_far + vertical_far)


def original_texture(image, rendered_size, target_fn=texture_energy):
    width, height = rendered_size
    luma = np.asarray(image.convert("L"), dtype=np.float32)
    result = []
    for x0, y0, x1, y1 in patch_boxes(rendered_size):
        left, top = x0 * image.width // width, y0 * image.height // height
        right = (x1 * image.width + width - 1) // width
        bottom = (y1 * image.height + height - 1) // height
        result.append(target_fn(luma[top:bottom, left:right]))
    return result


def training_assignments(sample):
    assignment = sha256(f"canonical-augmentation:{sample.image_id}".encode()).digest()
    return assignment[5:], assignment


def training_views(samples, table, deadline, *, target_fn=texture_energy, target_count=TARGET_COUNT,
                   record_view=None):
    targets = np.zeros((2, len(samples), len(TRAIN_CROPS), target_count), dtype=np.float32)
    active = np.zeros((2, len(samples)), dtype=bool)
    positions = {}
    for index, sample in enumerate(samples):
        for view, key in enumerate(training_assignments(sample)):
            if key[0] % 4 == 3:
                positions[key] = view, index
                active[view, index] = True
    if len(positions) != int(active.sum()):
        raise ValueError("Expected one target slot for every resize-and-JPEG training view")
    targets[active] = np.nan

    def acquire_with_target(image, assignment):
        processed = acquisition(image, assignment)
        if record_view is not None:
            record_view(processed.size, assignment)
        if assignment in positions:
            targets[positions[assignment]] = original_texture(image, processed.size, target_fn)
        return processed

    views = canonical_views(samples, table, deadline, paired=True, alter_both=True, acquisition=acquire_with_target)
    if not np.isfinite(targets).all():
        raise ValueError("Missing or invalid texture targets")
    return tuple(zip(views, map(torch.from_numpy, targets), map(torch.from_numpy, active), strict=True))


def texture_batch(view, ids, device):
    return tuple(batch_view(item, ids, device) for item in view)


def joint_loss(model, first, second, labels, *, auxiliary_weight=AUXILIARY_WEIGHT):
    pixels, target, active = (torch.cat((a, b)) for a, b in zip(first, second, strict=True))
    logits, reconstruction = model.classify_and_predict_texture(pixels)
    a, b = logits.float().chunk(2)
    return consistency_loss(a, b, labels) + auxiliary_weight * texture_loss(reconstruction, target, active)


def texture_loss(reconstruction, target, active):
    errors = F.smooth_l1_loss(reconstruction.float(), target, reduction="none").mean(dim=(1, 2))
    return (errors * active).sum() / active.sum().clamp_min(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("energy", "autocorrelation", "directional"), default="energy")
    parser.add_argument("--auxiliary-weight", type=float, default=AUXILIARY_WEIGHT)
    parser.add_argument("--phase-stem", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.auxiliary_weight) or args.auxiliary_weight < 0:
        parser.error("--auxiliary-weight must be finite and nonnegative")
    if args.phase_stem and args.target != "autocorrelation":
        parser.error("--phase-stem requires the autocorrelation target")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 85 * 60
    split, details = labeled_data()
    train, calibration = training_split(split["train"])
    tables, _ = tables_by_frequency(split["train"])
    seed_all(SEED, device)
    variant, target_fn, target_count, target_description = {
        "energy": (MODEL, texture_energy, TARGET_COUNT, "pre-acquisition log-RMS luminance differences at two scales"),
        "autocorrelation": (AUTOCORRELATION_MODEL, residual_autocorrelation, TARGET_COUNT,
                            "pre-acquisition luminance residual autocorrelation at pixel lags one and two"),
        "directional": (DIRECTIONAL_MODEL, directional_residual_autocorrelation, DIRECTIONAL_TARGET_COUNT,
                        "pre-acquisition horizontal and vertical luminance residual autocorrelation at lags one and two"),
    }[args.target]
    if args.phase_stem:
        variant = PHASE_STEM_MODEL
        model = PhaseStemTexture()
    else:
        model = SelfSupervisedTexture(target_count=target_count)
    model = model.to(device, memory_format=torch.channels_last)
    report = {"variant": variant, "seed": SEED, "epochs": 24, "split_sha256": details["split_sha256"],
              "gpu": torch.cuda.get_device_name(device), "train_count": len(train),
              "calibration_count": len(calibration), "auxiliary_weight": args.auxiliary_weight,
              "auxiliary_target": target_description, "auxiliary_target_count": target_count,
              "stem": "pixel_unshuffle" if args.phase_stem else "stride_two",
              "parameters": sum(param.numel() for param in model.parameters()), "stage": "training"}
    summary = args.output / "summary.json"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    views = training_views(train, tables[0], deadline, target_fn=target_fn, target_count=target_count)
    report["auxiliary_view_counts"] = [int(active.sum()) for _, _, active in views]
    print(f"Resize-and-JPEG views with texture targets: {report['auxiliary_view_counts']}", flush=True)
    fit(model, views, torch.tensor([row.label for row in train]), args.output, device, 24, deadline,
        variant=variant, get_batch=texture_batch, loss_fn=partial(joint_loss, auxiliary_weight=args.auxiliary_weight))
    report["stage"] = "training_complete"
    summary.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

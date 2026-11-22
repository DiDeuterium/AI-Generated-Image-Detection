"""Compare rich-texture and flat-region residuals within the same image."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from detection.data import jpeg_roundtrip, read_test
from detection.training import write_csv, write_json

from .cnn import device_for, run_lock
from .deep_fusion import CONFIRM, MIN_CONFIRM_GAIN, MIN_SCREEN_GAIN, SCREEN, bootstrap_mean_gain, combined_reference, comparison
from .group_patch import ARCHITECTURE
from .native import pad_to_crop
from .night_train import CROP, SEEDS, PatchCNN, cache as native_cache, check_time, fit, sample_rows, scores
from .robust import labeled_splits


VARIANT = "texture_contrast"
QUALITY = 75
TILE = CROP // 4
GRID = 7
TILES_PER_VIEW = 16
SCREEN_EPOCHS = 16
FULL_EPOCHS = 20


class TextureContrastCNN(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant != VARIANT:
            raise ValueError(f"Unexpected variant {variant}")
        self.encoder = PatchCNN(ARCHITECTURE).features
        self.classifier = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 96), nn.ReLU(),
                                        nn.Dropout(0.2), nn.Linear(96, 2))

    def forward(self, views):
        pixels = views.flatten(0, 1).float() / 255
        fine = pixels - F.avg_pool2d(pixels, 3, stride=1, padding=1, count_include_pad=False)
        broad = pixels - F.avg_pool2d(pixels, 5, stride=1, padding=2, count_include_pad=False)
        residuals = torch.cat((6 * fine, 4 * (broad - fine)), dim=1)
        mapped = self.encoder(residuals.contiguous(memory_format=torch.channels_last)).flatten(2)
        features = torch.cat((mapped.mean(dim=2), mapped.std(dim=2, correction=0)), dim=1)
        rich, poor = features.reshape(len(views), 2, -1).unbind(dim=1)
        contrast = torch.cat((rich - poor, (rich + poor) * 0.5), dim=1)
        return self.classifier(contrast)


def texture_views(image):
    image = pad_to_crop(image, CROP)
    x_positions = np.rint(np.linspace(0, image.width - TILE, GRID)).astype(int)
    y_positions = np.rint(np.linspace(0, image.height - TILE, GRID)).astype(int)
    pixels = np.asarray(image)
    tiles = np.stack([pixels[y:y + TILE, x:x + TILE] for y in y_positions for x in x_positions])
    rgb = tiles.astype(np.int32)
    light = (77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2]) // 256
    texture = (np.abs(light[:, :, 1:] - light[:, :, :-1]).mean(axis=(1, 2))
               + np.abs(light[:, 1:, :] - light[:, :-1, :]).mean(axis=(1, 2))
               + np.abs(light[:, 1:, 1:] - light[:, :-1, :-1]).mean(axis=(1, 2))
               + np.abs(light[:, 1:, :-1] - light[:, :-1, 1:]).mean(axis=(1, 2)))
    order = np.argsort(texture, kind="stable")

    def assemble(indices):
        mosaic = tiles[indices].reshape(4, 4, TILE, TILE, 3).transpose(0, 2, 1, 3, 4)
        image = Image.fromarray(mosaic.reshape(CROP, CROP, 3))
        decoded = jpeg_roundtrip(image, quality=QUALITY, subsampling=2)
        return np.asarray(decoded).transpose(2, 0, 1)

    return np.stack((assemble(order[-TILES_PER_VIEW:][::-1]), assemble(order[:TILES_PER_VIEW])))


def prepare(sample):
    with Image.open(sample.path) as source:
        return source.size, texture_views(source.convert("RGB"))


def cache(samples, deadline):
    print(f"Building rich/flat texture pairs for {len(samples)} images", flush=True)
    packed = np.empty((len(samples), 2, 3, CROP, CROP), dtype=np.uint8)
    sizes = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(samples), 32):
            check_time(deadline)
            for i, (size, views) in enumerate(executor.map(prepare, samples[start:start + 32]), start=start):
                packed[i] = views
                sizes.append(size)
            done = min(start + 32, len(samples))
            if done % 2048 == 0 or done == len(samples):
                print(f"Prepared {done}/{len(samples)}", flush=True)
    return torch.from_numpy(packed), sizes


def run(args, device, deadline):
    split_args = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    split, identity = labeled_splits(split_args, {"train_per_class": 3200, "val_per_class": 800,
                                                 "short_side": CROP, "crop_size": CROP})
    samples = split["train"] + split["val"]
    n_train = len(split["train"])
    labels = torch.tensor([sample.label for sample in samples])
    report = {"initialization": "random", "seeds": SEEDS, "screen_epochs": SCREEN_EPOCHS,
              "full_epochs": FULL_EPOCHS, "quality": QUALITY, "split_sha256": identity["split_sha256"],
              "reference": "56.5% spatial/spectrum rule", "screen": {},
              "gate": {"screen_mean_accuracy_gain": MIN_SCREEN_GAIN, "confirmation_gain": MIN_CONFIRM_GAIN,
                       "positive_auroc_and_accuracy_in_each_screen_cohort": True,
                       "positive_screen_bootstrap_lower_bound": True}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("prepare_labeled")
    paired, sizes = cache(samples, deadline)
    local_val, _ = native_cache(split["val"], QUALITY, deadline)
    real = sorted((i for i in range(n_train, len(samples)) if samples[i].label == 0),
                  key=lambda i: sha256(f"threshold-audit:{samples[i].image_id}".encode()).digest())
    offset = 0
    comparisons = []

    def train_models(directory, rows, epochs):
        return [fit(directory / f"s{seed}", VARIANT, seed, epochs, paired, labels, rows, device, deadline,
                    model_factory=TextureContrastCNN) for seed in SEEDS]

    def fold(cohort, epochs):
        nonlocal offset
        training, group = sample_rows(samples, sizes, n_train, cohort)
        fake = group[len(group) // 2:].tolist()
        rows = torch.tensor(real[offset:offset + len(fake)] + fake)
        if len(rows) != 2 * len(fake):
            raise ValueError("Insufficient disjoint real validation samples")
        offset += len(fake)
        models = train_models(args.output / f"cohort_{cohort}", training, epochs)
        candidate = scores(models, paired, rows, device, deadline)
        if cohort == CONFIRM:
            local, _ = native_cache(samples, QUALITY, deadline)
            reference = combined_reference(args, cohort, local, labels, training, rows, device, deadline)
        else:
            reference = combined_reference(args, cohort, local_val, labels[n_train:], training,
                                           rows - n_train, device, deadline)
        truth = labels[rows].numpy()
        result = comparison(truth, reference, candidate)
        result["texture_contrast"] = result.pop("deep_fusion")
        if cohort in SCREEN:
            comparisons.append((truth, reference, candidate))
        print(f"Cohort {cohort}: accuracy {result['reference']['accuracy']:.3%} -> "
              f"{result['texture_contrast']['accuracy']:.3%}; AUROC {result['reference']['auroc']:.4f} -> "
              f"{result['texture_contrast']['auroc']:.4f}", flush=True)
        return result

    for cohort in SCREEN:
        record(f"screen_{cohort}")
        report["screen"][str(cohort)] = fold(cohort, SCREEN_EPOCHS)
    grouped = bootstrap_mean_gain(comparisons)
    report["screen_paired_summary"] = grouped
    improved = all(all(change > 0 for change in item["delta"].values()) for item in report["screen"].values())
    if not improved or grouped["mean_gain"] < MIN_SCREEN_GAIN or grouped["bootstrap_95_percent"][0] <= 0:
        record("stopped_at_screen")
        print(f"Submission gate failed at screening: {grouped}", flush=True)
        return

    record("confirm_128")
    confirm = fold(CONFIRM, FULL_EPOCHS)
    report["confirmation"] = confirm
    if confirm["delta"]["accuracy"] < MIN_CONFIRM_GAIN or confirm["delta"]["auroc"] <= 0:
        record("stopped_at_confirmation")
        print("Submission gate failed on the independent confirmation cohort", flush=True)
        return

    record("full_training")
    training, _ = sample_rows(samples, sizes, n_train)
    models = train_models(args.output / "full", training, FULL_EPOCHS)
    report["train_count"] = len(training)
    del local_val, paired
    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"),
                     Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    views, _ = cache(test, deadline)
    predicted = scores(models, views, torch.arange(len(test)), device, deadline) > 0.5
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(label)) for sample, label in zip(test, predicted, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

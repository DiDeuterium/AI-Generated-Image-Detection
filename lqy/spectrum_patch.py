"""Run a paired RGB-to-spectrum input ablation on labeled cohort holdouts."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from detection.data import read_test
from detection.metrics import binary_metrics
from detection.training import write_csv, write_json

from .cnn import device_for, run_lock
from .group_patch import ARCHITECTURE, restore_ensemble
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, PatchCNN, check_time, fit, prepare_image, sample_rows, scores
from .robust import labeled_splits


VARIANT = "spectrum_and_residual"
SCREEN = (512, 1024, 256)
CONFIRM = 128
QUALITY = 75

axis = np.arange(CROP) - CROP // 2
y, x = np.meshgrid(axis, axis, indexing="ij")
radius = np.hypot(x, y)
period = CROP // 8
grid = (np.abs((x + period // 2) % period - period // 2) <= 1
        ) & (np.abs((y + period // 2) % period - period // 2) <= 1)
mask = grid | (radius < 3)
bands = (radius // 2).astype(np.intp).ravel()
window = np.outer(np.hanning(CROP), np.hanning(CROP)).astype(np.float32)
unmasked = np.flatnonzero(~mask.ravel())
band_order = unmasked[np.argsort(bands[unmasked], kind="stable")]
active_bands, band_starts, band_sizes = np.unique(bands[band_order], return_index=True, return_counts=True)
midrange = (active_bands >= 5) & (active_bands <= 24)


class SpectrumPatchCNN(PatchCNN):
    def __init__(self, variant):
        if variant != VARIANT:
            raise ValueError(f"Unknown spectrum variant {variant}")
        super().__init__(ARCHITECTURE)

    def patch_features(self, views):
        _, residual = self.rgb_and_residual(views[:, :, :3])
        spectrum = views[:, :, 3:].flatten(0, 1).float() / 255 - 0.5
        return torch.cat((spectrum, residual), dim=1)


def spectrum_channels(patches):
    pixels = patches.astype(np.float32) / 255
    pixels -= pixels.mean(axis=(-2, -1), keepdims=True)
    frequency = np.fft.fftshift(np.fft.fft2(pixels * window, axes=(-2, -1)), axes=(-2, -1))
    log_amplitude = np.log(np.maximum(np.abs(frequency) / CROP, 1e-4)).reshape(-1, CROP * CROP)
    radial = np.add.reduceat(log_amplitude[:, band_order], band_starts, axis=1) / band_sizes
    radial_map = np.zeros((len(log_amplitude), int(bands.max()) + 1), dtype=log_amplitude.dtype)
    radial_map[:, active_bands] = radial
    average = radial[:, midrange].mean(axis=1, keepdims=True)
    expected = radial_map[:, bands]
    # Keep directional peaks and a smaller contribution from the radial slope.
    normalized = log_amplitude - expected + 0.35 * (expected - average)
    normalized[:, mask.ravel()] = 0
    encoded = np.rint(np.clip(normalized / 6 + 0.5, 0, 1) * 255).astype(np.uint8)
    return encoded.reshape(patches.shape)


def prepare(sample):
    size, patches = prepare_image(sample, QUALITY)
    return size, np.concatenate((patches, spectrum_channels(patches)), axis=1)


def cache(samples, deadline):
    print(f"Preparing Q{QUALITY} RGB and spectra for {len(samples)} images", flush=True)
    views = np.empty((len(samples), 5, 6, CROP, CROP), dtype=np.uint8)
    sizes = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(samples), 32):
            check_time(deadline)
            block = executor.map(prepare, samples[start:start + 32])
            for index, (size, pixels) in enumerate(block, start=start):
                sizes.append(size)
                views[index] = pixels
            completed = min(start + 32, len(samples))
            if completed % 1024 == 0 or completed == len(samples):
                print(f"Prepared {completed}/{len(samples)}", flush=True)
    return torch.from_numpy(views), sizes


def accuracy_interval(truth, first, second):
    paired = (second > 0.5) == truth
    change = paired.astype(np.int8) - (((first > 0.5) == truth).astype(np.int8))
    rng = np.random.default_rng(17)
    estimates = np.zeros(2000)
    for label in (0, 1):
        subset = change[truth == label]
        draw = rng.integers(len(subset), size=(len(estimates), len(subset)))
        estimates += subset[draw].sum(axis=1) / len(change)
    return {"after_only_correct": int((change == 1).sum()), "before_only_correct": int((change == -1).sum()),
            "bootstrap_95_percent": np.quantile(estimates, (0.025, 0.975)).tolist()}


def compare(truth, reference, candidate):
    a = binary_metrics(truth, reference > 0.5, reference)
    b = binary_metrics(truth, candidate > 0.5, candidate)
    delta = {name: b[name] - a[name] for name in ("accuracy", "auroc")}
    return {"reference": a, "spectrum": b, "delta": delta,
            "paired_accuracy": accuracy_interval(truth, reference, candidate)}


def run(args, device, deadline):
    split_args = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    split, identity = labeled_splits(split_args, {"train_per_class": 3200, "val_per_class": 800,
                                                  "short_side": CROP, "crop_size": CROP})
    samples = split["train"] + split["val"]
    train_end = len(split["train"])
    labels = torch.tensor([sample.label for sample in samples])
    report = {"screen_cohorts": SCREEN, "confirmation_cohort": CONFIRM, "jpeg_quality": QUALITY,
              "seeds": SEEDS, "epochs": SCREEN_EPOCHS, "split_sha256": identity["split_sha256"],
              "real_validation_samples_disjoint": True,
              "gate": "At least two screen cohorts improve both metrics, then 128 improves both",
              "results": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("prepare_labeled")
    views, sizes = cache(samples, deadline)
    rgb = views[:, :, :3]
    real = sorted((i for i in range(train_end, len(samples)) if samples[i].label == 0),
                  key=lambda i: sha256(f"threshold-audit:{samples[i].image_id}".encode()).digest())
    offset = 0

    def train_models(folder, rows, spectral=True):
        model_class = SpectrumPatchCNN if spectral else PatchCNN
        variant = VARIANT if spectral else ARCHITECTURE
        data = views if spectral else rgb
        return [fit(folder / f"s{seed}", variant, seed, SCREEN_EPOCHS, data, labels, rows, device,
                    deadline, model_factory=model_class) for seed in SEEDS]

    def fold(cohort):
        nonlocal offset
        train_rows, group = sample_rows(samples, sizes, train_end, cohort)
        fake = group[len(group) // 2:].tolist()
        ensemble = train_models(args.output / f"spectrum_{cohort}", train_rows)
        validation = torch.tensor(real[offset:offset + len(fake)] + fake)
        if len(validation) != 2 * len(fake):
            raise ValueError("Insufficient disjoint real validation samples")
        offset += len(fake)
        if cohort == CONFIRM:
            reference = train_models(args.output / "reference_128", train_rows, spectral=False)
        else:
            root = (args.group_reference / "confirm_256_reference" if cohort == 256
                    else args.night_reference / f"screen_{cohort}_{ARCHITECTURE}")
            reference = restore_ensemble([root / f"s{seed}" / "last.pt" for seed in SEEDS], device)
        first = scores(reference, rgb, validation, device, deadline)
        second = scores(ensemble, views, validation, device, deadline)
        result = compare(labels[validation].numpy(), first, second)
        for name in ("reference", "spectrum"):
            current = result[name]
            print(f"{cohort} {name}: accuracy={current['accuracy']:.3%} AUROC={current['auroc']:.4f}", flush=True)
        return result

    record("screening")
    for cohort in SCREEN:
        report["results"][str(cohort)] = fold(cohort)
        record("screening")
    wins = sum(all(delta > 0 for delta in item["delta"].values()) for item in report["results"].values())
    report["screen_cohorts_improving_both"] = wins
    if wins < 2:
        record("stopped_at_screen")
        return

    record("confirmation")
    report["confirmation"] = fold(CONFIRM)
    if any(delta <= 0 for delta in report["confirmation"]["delta"].values()):
        record("stopped_at_confirmation")
        return

    record("full_training")
    train_rows, val_rows = sample_rows(samples, sizes, train_end)
    ensemble = train_models(args.output / "full_spectrum", train_rows)
    validation = scores(ensemble, views, val_rows, device, deadline)
    report["full_validation"] = binary_metrics(labels[val_rows].tolist(), validation > 0.5, validation)
    del rgb, views

    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"),
                     Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    packed, _ = cache(test, deadline)
    prediction = scores(ensemble, packed, torch.arange(len(test)), device, deadline) > 0.5
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(label)) for sample, label in zip(test, prediction, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--group-reference", type=Path, default=Path("lqy/runs/group-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 1800)


if __name__ == "__main__":
    main()

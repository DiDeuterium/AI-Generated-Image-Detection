"""Screen a scratch CNN on brightness-conditioned image noise curves."""

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

from detection.data import read_test
from detection.training import load_checkpoint, save_checkpoint, seed_all, write_csv, write_json

from .cnn import run_lock
from .deep_fusion import CONFIRM, MIN_CONFIRM_GAIN, MIN_SCREEN_GAIN, SCREEN, bootstrap_mean_gain, combined_reference, comparison
from .night_train import CROP, SEEDS, cache as patch_cache, check_time, image_patches, sample_rows
from .robust import labeled_splits


BRIGHTNESS = np.arange(16, 256, 32)
QUALITY = (75, 90)


def square_tiles(patches):
    return patches.reshape(-1, 3, CROP // 8, 8, CROP // 8, 8).transpose(0, 2, 4, 1, 3, 5).reshape(-1, 3, 8, 8)


def noise_curve(patches):
    tiles = square_tiles(patches.astype(np.float32))
    luma = (0.299 * tiles[:, 0] + 0.587 * tiles[:, 1] + 0.114 * tiles[:, 2])
    tiles[:, 0] -= luma
    tiles[:, 2] -= luma
    tiles[:, 1] = luma

    # Coarse gradients guide smooth-area selection. Diagonal Haar detail is
    # measured within 8px blocks to avoid cross-block JPEG edges.
    coarse = luma.reshape(-1, 4, 2, 4, 2).mean(axis=(2, 4))
    structure = (np.abs(np.diff(coarse, axis=1)).mean(axis=(1, 2))
                 + np.abs(np.diff(coarse, axis=2)).mean(axis=(1, 2)))
    detail = (tiles[:, :, ::2, ::2] - tiles[:, :, 1::2, ::2]
              - tiles[:, :, ::2, 1::2] + tiles[:, :, 1::2, 1::2]) * 0.5
    var = np.mean(detail ** 2, axis=(2, 3))
    luma_var = var[:, 1]
    corr = np.stack([
        (detail[:, 1] * detail[:, channel]).mean(axis=(1, 2)) / np.sqrt((0.25 + luma_var) * (0.25 + var[:, channel]))
        for channel in (0, 2)
    ], axis=1)
    same_axis = (detail[:, 1, :, 1:] * detail[:, 1, :, :-1]).mean(axis=(1, 2))
    other_axis = (detail[:, 1, 1:, :] * detail[:, 1, :-1, :]).mean(axis=(1, 2))
    autocorr = (same_axis + other_axis) / (2 * (0.25 + luma_var))

    measurements = np.column_stack((np.log1p(var), corr, autocorr))
    bin_index = np.minimum((luma.mean(axis=(1, 2)) / 32).astype(int), len(BRIGHTNESS) - 1)
    curve_rows = np.zeros((len(BRIGHTNESS), 2 * measurements.shape[1]), dtype=np.float32)
    present = []
    for index in range(len(BRIGHTNESS)):
        inside = np.flatnonzero(bin_index == index)
        if len(inside) < 12:
            continue
        order = np.argsort(structure[inside], kind="stable")
        flat = inside[order[:max(8, len(inside) // 3)]]
        curve_rows[index] = np.quantile(measurements[flat], (0.5, 0.8), axis=0).ravel()
        present.append(index)
    if present:
        for channel in range(curve_rows.shape[1]):
            curve_rows[:, channel] = np.interp(BRIGHTNESS, BRIGHTNESS[present], curve_rows[present, channel])
    else:
        curve_rows[:] = np.tile(np.median(measurements, axis=0), 2)
    return curve_rows.T.copy()


def prepare(sample):
    with Image.open(sample.path) as source:
        size = source.size
        rgb = source.convert("RGB")
    return size, np.stack([noise_curve(image_patches(rgb, quality)) for quality in QUALITY])


def cache(samples, deadline):
    data, sizes = [], []
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(samples), 32):
            check_time(deadline)
            for size, features in workers.map(prepare, samples[start:start + 32]):
                sizes.append(size)
                data.append(features)
            if start % 1600 == 0:
                print(f"Computed noise curves for {len(data)}/{len(samples)} images", flush=True)
    return torch.from_numpy(np.stack(data)), sizes


class CurveCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.features = nn.Sequential(nn.Conv1d(n_features, 24, 3, padding=1), nn.GELU(),
                                      nn.Conv1d(24, 24, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(24 * len(BRIGHTNESS), 2))

    def forward(self, data):
        return self.head(self.features(data))


def fit(data, labels, train_rows, folder, deadline):
    training = data[train_rows].flatten(0, 1)
    mean = training.mean(dim=(0, 2), keepdim=True)
    std = training.std(dim=(0, 2), keepdim=True).clamp(min=0.05)
    x = ((training - mean) / std).clamp(-8, 8)
    y = labels[train_rows].repeat_interleave(data.shape[1])
    models = []
    for seed in SEEDS:
        seed_all(seed, torch.device("cpu"))
        model = CurveCNN(training.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(16):
            check_time(deadline)
            for rows in torch.randperm(len(y), generator=generator).split(128):
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(x[rows]), y[rows]).backward()
                optimizer.step()
        models.append(model.eval())
        save_checkpoint(folder / f"seed-{seed}.pt", {"weights": model.state_dict(), "mean": mean, "std": std})
    return models, mean, std


@torch.inference_mode()
def view_probabilities(state, data):
    models, mean, std = state
    x = ((data.flatten(0, 1) - mean) / std).clamp(-8, 8)
    return torch.stack([model(x).softmax(dim=1)[:, 1] for model in models]).mean(dim=0).reshape(len(data), -1).numpy()


def probabilities(state, data):
    return view_probabilities(state, data).mean(axis=1)


def restore(folder):
    models, shared = [], None
    for seed in SEEDS:
        state = load_checkpoint(folder / f"seed-{seed}.pt")
        mean, std = state["mean"], state["std"]
        if shared is not None and not all(torch.equal(a, b) for a, b in zip(shared, (mean, std), strict=True)):
            raise ValueError(f"Incompatible noise curve normalization: {folder}")
        shared = mean, std
        model = CurveCNN(mean.shape[1])
        model.load_state_dict(state["weights"])
        models.append(model.eval())
    return models, *shared


def run(args, deadline):
    source = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    split, identity = labeled_splits(source, {"train_per_class": 3200, "val_per_class": 800,
                                             "short_side": CROP, "crop_size": CROP})
    samples = split["train"] + split["val"]
    train_end = len(split["train"])
    labels = torch.tensor([sample.label for sample in samples])
    report = {"stage": "prepare_labeled", "split_sha256": identity["split_sha256"], "screen": {}, "quality": QUALITY}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("prepare_labeled")
    data, sizes = cache(samples, deadline)
    real_rows = sorted((i for i in range(train_end, len(samples)) if samples[i].label == 0),
                       key=lambda i: sha256(f"threshold-audit:{samples[i].image_id}".encode()).digest())
    offset, paired = 0, []
    cpu = torch.device("cpu")

    def compare_group(cohort):
        nonlocal offset
        train_rows, group = sample_rows(samples, sizes, train_end, cohort)
        fake_rows = group[len(group) // 2:].tolist()
        rows = torch.tensor(real_rows[offset:offset + len(fake_rows)] + fake_rows)
        offset += len(fake_rows)
        model = fit(data, labels, train_rows, args.output / f"cohort-{cohort}", deadline)
        current = probabilities(model, data[rows])
        if cohort in SCREEN:
            patches, _ = patch_cache([samples[i] for i in rows.tolist()], QUALITY[0], deadline)
            previous = combined_reference(args, cohort, patches, labels, train_rows, torch.arange(len(rows)), cpu, deadline)
        else:
            packed_rows = torch.cat((train_rows, rows))
            patches, _ = patch_cache([samples[i] for i in packed_rows.tolist()], QUALITY[0], deadline)
            previous = combined_reference(args, cohort, patches, labels[packed_rows], torch.arange(len(train_rows)),
                                          torch.arange(len(train_rows), len(packed_rows)), cpu, deadline)
        truth = labels[rows].numpy()
        result = comparison(truth, previous, current)
        result["noise_curve"] = result.pop("deep_fusion")
        print(f"{cohort}: accuracy {result['reference']['accuracy']:.3%} -> {result['noise_curve']['accuracy']:.3%}, "
              f"AUROC {result['reference']['auroc']:.4f} -> {result['noise_curve']['auroc']:.4f}", flush=True)
        paired.append((truth, previous, current))
        return result

    for cohort in SCREEN:
        record(f"screen_{cohort}")
        report["screen"][str(cohort)] = compare_group(cohort)
    grouped = bootstrap_mean_gain(paired)
    report["screen_paired_summary"] = grouped
    improved = all(all(delta > 0 for delta in row["delta"].values()) for row in report["screen"].values())
    if not improved or grouped["mean_gain"] < MIN_SCREEN_GAIN or grouped["bootstrap_95_percent"][0] <= 0:
        record("stopped_at_screen")
        return

    record("confirm_128")
    confirmation = compare_group(CONFIRM)
    report["confirmation"] = confirmation
    if confirmation["delta"]["accuracy"] < MIN_CONFIRM_GAIN or confirmation["delta"]["auroc"] <= 0:
        record("stopped_at_confirmation")
        return

    record("full_training")
    model = fit(data, labels, torch.arange(train_end), args.output / "full", deadline)
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    features, _ = cache(test, deadline)
    predicted = (probabilities(model, features) > 0.5).astype(int)
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              zip((sample.image_id for sample in test), predicted, strict=True))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    torch.set_num_threads(3)
    with run_lock(args.output):
        run(args, time.monotonic() + 3600)


if __name__ == "__main__":
    main()

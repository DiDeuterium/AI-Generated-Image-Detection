"""Train a local texture and full-image ResNet detector from random weights."""

import argparse
import math
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
from torchvision.models import resnet18

from detection.data import jpeg_roundtrip, read_test
from detection.metrics import binary_metrics
from detection.training import load_checkpoint, save_checkpoint, seed_all, write_csv, write_json

from .cnn import device_for, run_lock
from .group_patch import ARCHITECTURE, restore_ensemble
from .night_train import BATCH, CROP, SCREEN_EPOCHS as REFERENCE_EPOCHS, SEEDS, PatchCNN, check_time
from .night_train import fit as fit_reference
from .night_train import image_patches, sample_rows, scores
from .robust import labeled_splits
from .spectrum_patch import VARIANT as SPECTRUM_VARIANT
from .spectrum_patch import SpectrumPatchCNN, accuracy_interval, spectrum_channels


QUALITY = 75
SCREEN = (512, 1024)
CONFIRM = 128
SCREEN_EPOCHS = 12
FULL_EPOCHS = 24
MIN_SCREEN_GAIN = 0.05
MIN_CONFIRM_GAIN = 0.03
MODEL = "resnet18-and-native-patches"


class DeepFusionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = PatchCNN(ARCHITECTURE)
        self.scene = resnet18(weights=None)
        self.scene.fc = nn.Identity()
        self.local_projection = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.ReLU())
        self.scene_projection = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 128), nn.ReLU())
        self.scene_classifier = nn.Linear(128, 2)
        self.fusion = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128), nn.ReLU(),
                                    nn.Dropout(0.2), nn.Linear(128, 2))

    def heads(self, views):
        pixels = self.local.patch_features(views[:, :5])
        features = self.local.features(pixels.contiguous(memory_format=torch.channels_last)).flatten(2)
        patch = torch.cat((features.mean(dim=2), features.std(dim=2, correction=0)), dim=1)
        patch = patch.reshape(len(views), 5, -1)
        local = self.local_projection(torch.cat((patch.mean(dim=1), patch.std(dim=1, correction=0)), dim=1))
        entire = (views[:, 5].float() / 255 - 0.5).contiguous(memory_format=torch.channels_last)
        whole = self.scene_projection(self.scene(entire))
        joint = self.fusion(torch.cat((local, whole), dim=1))
        return joint, self.local.classifier(patch).mean(dim=1), self.scene_classifier(whole)

    def forward(self, views):
        return self.heads(views)[0]


def prepare(sample):
    with Image.open(sample.path) as source:
        size = source.size
        rgb = source.convert("RGB")
    local = image_patches(rgb, QUALITY)
    whole = rgb.resize((CROP, CROP), Image.Resampling.LANCZOS)
    compressed = jpeg_roundtrip(whole, quality=QUALITY, subsampling=2)
    return size, np.concatenate((local, np.asarray(compressed).transpose(2, 0, 1)[None]), axis=0)


def cache(samples, deadline):
    print(f"Preparing six views per image for {len(samples)} images", flush=True)
    packed = np.empty((len(samples), 6, 3, CROP, CROP), dtype=np.uint8)
    sizes = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(samples), 32):
            check_time(deadline)
            for i, (size, views) in enumerate(executor.map(prepare, samples[start:start + 32]), start=start):
                packed[i] = views
                sizes.append(size)
    return torch.from_numpy(packed), sizes


def learning_rate(epoch, epochs):
    if epoch <= 2:
        return 3e-4 * epoch / 2
    return 3e-5 + (3e-4 - 3e-5) * (1 + math.cos(math.pi * (epoch - 2) / (epochs - 2))) / 2


def fit(output, seed, epochs, packed, labels, train, validation, device, deadline, identity):
    seed_all(seed, device)
    model = DeepFusionCNN().to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    use_amp = device.type in ("xpu", "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp, init_scale=4096)
    path = output / "last.pt"
    meta = {"model": MODEL, "seed": seed, "epochs": epochs, "split": identity,
            "train_rows_sha256": sha256(train.numpy().tobytes()).hexdigest()}
    history = []
    if path.is_file():
        state = load_checkpoint(path)
        if state["metadata"] != meta:
            raise ValueError(f"Checkpoint does not match this run: {path}")
        model.load_state_dict(state["weights"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]

    for epoch in range(len(history) + 1, epochs + 1):
        started = time.monotonic()
        epoch_seed = seed * 10000 + epoch
        seed_all(epoch_seed, device)
        random = torch.Generator().manual_seed(epoch_seed)
        optimizer.param_groups[0]["lr"] = learning_rate(epoch, epochs)
        running = torch.zeros((), device=device)
        model.train()
        for indices in train[torch.randperm(len(train), generator=random)].split(BATCH):
            check_time(deadline)
            images = packed[indices]
            flip = torch.rand(len(indices), generator=random) < 0.5
            images[flip] = images[flip].flip(-1)
            images = images.to(device)
            target = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                main, local, scene = model.heads(images)
                loss = F.cross_entropy(main, target) + 0.2 * (F.cross_entropy(local, target)
                                                               + F.cross_entropy(scene, target))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.detach() * len(indices)
        mean_loss = running.item() / len(train)
        if not np.isfinite(mean_loss):
            raise ValueError(f"Non-finite loss during epoch {epoch}")
        entry = {"epoch": epoch, "loss": mean_loss, "seconds": time.monotonic() - started}
        if epoch % 4 == 0 or epoch == epochs:
            probability = scores([model], packed, validation, device, deadline)
            entry["validation"] = binary_metrics(labels[validation].tolist(), probability > 0.5, probability)
        history.append(entry)
        save_checkpoint(path, {"metadata": meta, "weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "scaler": scaler.state_dict(), "history": history})
        write_json(output / "epochs.json", history)
        validation_text = (f" val={entry['validation']['accuracy']:.3%} AUC={entry['validation']['auroc']:.4f}"
                           if "validation" in entry else "")
        print(f"{output.parent.name}/{output.name} epoch {epoch}/{epochs}: loss={mean_loss:.4f}, "
              f"{entry['seconds']:.1f}s{validation_text}", flush=True)
    return model


def spectral_views(local, rows, deadline):
    def encode(index):
        patches = local[index].numpy()
        return np.concatenate((patches, spectrum_channels(patches)), axis=1)

    data = np.empty((len(rows), 5, 6, CROP, CROP), dtype=np.uint8)
    indices = rows.tolist()
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(indices), 32):
            check_time(deadline)
            for i, views in enumerate(executor.map(encode, indices[start:start + 32]), start=start):
                data[i] = views
    return torch.from_numpy(data)


def combined_reference(args, cohort, local, labels, train, val, device, deadline, *, return_components=False):
    if cohort in SCREEN:
        path = args.night_reference / f"screen_{cohort}_{ARCHITECTURE}"
        spatial = restore_ensemble([path / f"s{seed}" / "last.pt" for seed in SEEDS], device)
        spectral = []
        for seed in SEEDS:
            checkpoint = args.spectrum_reference / f"spectrum_{cohort}" / f"s{seed}" / "last.pt"
            state = load_checkpoint(checkpoint)
            if ((state["variant"], state["seed"], state["epoch"])
                    != (SPECTRUM_VARIANT, seed, REFERENCE_EPOCHS)):
                raise ValueError(f"Incompatible reference checkpoint: {checkpoint}")
            model = SpectrumPatchCNN(SPECTRUM_VARIANT).to(device, memory_format=torch.channels_last)
            model.load_state_dict(state["weights"])
            spectral.append(model)
        spectral_data = spectral_views(local, val, deadline)
        spectral_rows = torch.arange(len(val))
    else:
        print("Rebuilding the 50/50 baseline on the independent 128 cohort", flush=True)
        reference = args.output / "reference_128"
        spatial = [fit_reference(reference / "spatial" / f"s{seed}", ARCHITECTURE, seed,
                                 REFERENCE_EPOCHS, local, labels, train, device, deadline) for seed in SEEDS]
        rows = torch.cat((train, val))
        spectral_data = spectral_views(local, rows, deadline)
        spectral = [fit_reference(reference / "spectrum" / f"s{seed}", SPECTRUM_VARIANT, seed,
                                  REFERENCE_EPOCHS, spectral_data, labels[rows], torch.arange(len(train)),
                                  device, deadline, model_factory=SpectrumPatchCNN) for seed in SEEDS]
        spectral_rows = torch.arange(len(train), len(rows))

    components = (scores(spatial, local, val, device, deadline),
                  scores(spectral, spectral_data, spectral_rows, device, deadline))
    if return_components:
        return components
    return 0.5 * (components[0] + components[1])


def comparison(truth, previous, current):
    before = binary_metrics(truth, previous > 0.5, previous)
    after = binary_metrics(truth, current > 0.5, current)
    return {"reference": before, "deep_fusion": after,
            "delta": {key: after[key] - before[key] for key in ("accuracy", "auroc")},
            "paired_accuracy": accuracy_interval(truth, previous, current)}


def bootstrap_mean_gain(folds):
    rng = np.random.default_rng(17)
    estimate = np.zeros(4000)
    gains = []
    for truth, previous, current in folds:
        better = ((current > 0.5) == truth).astype(np.int8)
        worse = ((previous > 0.5) == truth).astype(np.int8)
        difference = better - worse
        gains.append(float(difference.mean()))
        for label in (0, 1):
            group = difference[truth == label]
            sample = rng.integers(len(group), size=(len(estimate), len(group)))
            estimate += group[sample].sum(axis=1) / (len(folds) * len(difference))
    return {"mean_gain": sum(gains) / len(gains),
            "bootstrap_95_percent": np.quantile(estimate, (0.025, 0.975)).tolist()}


def run(args, device, deadline):
    data = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    split, identity = labeled_splits(data, {"train_per_class": 3200, "val_per_class": 800,
                                           "short_side": CROP, "crop_size": CROP})
    samples = split["train"] + split["val"]
    train_end = len(split["train"])
    labels = torch.tensor([sample.label for sample in samples])
    split_hash = identity["split_sha256"]
    report = {"model": MODEL, "initialization": "random", "screen_epochs": SCREEN_EPOCHS,
              "full_epochs": FULL_EPOCHS, "seeds": SEEDS, "split_sha256": split_hash, "screen": {},
              "reference": "56.5% spatial and spectrum ensemble, reconstructed on matching labeled folds",
              "gate": {"screen_min_mean_accuracy_gain": MIN_SCREEN_GAIN,
                       "screen_both_metrics_improve_in_each_cohort": True,
                       "screen_95_percent_lower_bound_must_be_positive": True,
                       "confirmation_cohort": CONFIRM, "confirmation_min_accuracy_gain": MIN_CONFIRM_GAIN,
                       "confirmation_auroc_must_improve": True}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("prepare_labeled")
    views, sizes = cache(samples, deadline)
    local = views[:, :5]
    real = sorted((i for i in range(train_end, len(samples)) if samples[i].label == 0),
                  key=lambda i: sha256(f"threshold-audit:{samples[i].image_id}".encode()).digest())
    used = 0
    paired = []

    def fold(cohort, epochs):
        nonlocal used
        train, group = sample_rows(samples, sizes, train_end, cohort)
        fake = group[len(group) // 2:].tolist()
        val = torch.tensor(real[used:used + len(fake)] + fake)
        if len(val) != 2 * len(fake):
            raise ValueError("Insufficient disjoint real validation samples")
        used += len(fake)
        folder = f"screen_{cohort}" if cohort in SCREEN else "confirm_128"
        model = fit(args.output / folder / f"s{SEEDS[0]}", SEEDS[0], epochs, views,
                    labels, train, val, device, deadline, split_hash)
        current = scores([model], views, val, device, deadline)
        previous = combined_reference(args, cohort, local, labels, train, val, device, deadline)
        truth = labels[val].numpy()
        result = comparison(truth, previous, current)
        if cohort in SCREEN:
            paired.append((truth, previous, current))
        print(f"Cohort {cohort}: accuracy {result['reference']['accuracy']:.3%} -> "
              f"{result['deep_fusion']['accuracy']:.3%}; AUROC {result['reference']['auroc']:.4f} -> "
              f"{result['deep_fusion']['auroc']:.4f}", flush=True)
        return result

    for cohort in SCREEN:
        record(f"screen_{cohort}")
        report["screen"][str(cohort)] = fold(cohort, SCREEN_EPOCHS)
    grouped = bootstrap_mean_gain(paired)
    report["screen_paired_summary"] = grouped
    improved = all(all(gain > 0 for gain in item["delta"].values()) for item in report["screen"].values())
    if (not improved or grouped["mean_gain"] < MIN_SCREEN_GAIN
            or grouped["bootstrap_95_percent"][0] <= 0):
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
    train, val = sample_rows(samples, sizes, train_end)
    full = []
    for seed in SEEDS:
        full.append(fit(args.output / "full" / f"s{seed}", seed, FULL_EPOCHS, views, labels, train, val,
                        device, deadline, split_hash))
    value = scores(full, views, val, device, deadline)
    report["full_validation"] = binary_metrics(labels[val].tolist(), value > 0.5, value)
    report["train_count"] = len(train)
    del local, views

    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"),
                     Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    views, _ = cache(test, deadline)
    predicted = scores(full, views, torch.arange(len(test)), device, deadline) > 0.5
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(label)) for sample, label in zip(test, predicted, strict=True)))
    record("complete")
    print(f"Wrote {len(test)} predictions", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 7200)


if __name__ == "__main__":
    main()

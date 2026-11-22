"""Screen scratch-trained patch CNNs on labeled cohort holdouts before a longer run."""

import argparse
import ctypes
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from detection.data import jpeg_roundtrip, read_test
from detection.metrics import binary_metrics
from detection.training import save_checkpoint, seed_all, write_csv, write_json

from .cnn import device_for, run_lock
from .native import pad_to_crop
from .robust import labeled_splits


CROP = 128
BATCH = 32
SEEDS = (17, 42)
SCREEN_COHORTS = (512, 1024)
CONFIRM_COHORT = 256
VARIANTS = ("rgb_and_residual", "residual")
SCREEN_EPOCHS = 6
LONG_EPOCHS = 16
SCREEN_MIN = 0.72
SCREEN_MEAN = 0.75
CONFIRM_MIN = 0.75


class PatchCNN(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.use_rgb = variant == "rgb_and_residual"
        self.features = nn.Sequential(
            nn.Conv2d(6 if self.use_rgb else 3, 16, 5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.classifier = nn.Linear(128, 2)

    @staticmethod
    def rgb_and_residual(views):
        rgb = views.flatten(0, 1).float() / 255 - 0.5
        blurred = F.avg_pool2d(rgb, 3, stride=1, padding=1, count_include_pad=False)
        return rgb, 6 * (rgb - blurred)

    def patch_features(self, views):
        rgb, residual = self.rgb_and_residual(views)
        return torch.cat((rgb, residual), dim=1) if self.use_rgb else residual

    def patch_embeddings(self, views):
        pixels = self.patch_features(views)
        mapped = self.features(pixels.contiguous(memory_format=torch.channels_last)).flatten(2)
        return torch.cat((mapped.mean(dim=2), mapped.std(dim=2, correction=0)), dim=1)

    def forward(self, views):
        pooled = self.patch_embeddings(views)
        return self.classifier(pooled).reshape(len(views), views.shape[1], 2).mean(dim=1)


def prepare_image(sample, quality, crop_size=CROP):
    with Image.open(sample.path) as source:
        size = source.size
        rgb = source.convert("RGB")
    return size, image_patches(rgb, quality, crop_size=crop_size)


def image_patches(rgb, quality, transform=None, *, encoder=None, crop_size=CROP):
    rgb = pad_to_crop(rgb, crop_size)
    right, bottom = rgb.width - crop_size, rgb.height - crop_size
    positions = ((right // 2, bottom // 2), (0, 0), (right, 0), (0, bottom), (right, bottom))
    images = []
    for left, top in positions:
        crop = rgb.crop((left, top, left + crop_size, top + crop_size))
        if transform is not None:
            crop = transform(crop)
        decoded = encoder(crop) if encoder is not None else jpeg_roundtrip(crop, quality=quality, subsampling=2)
        images.append(np.asarray(decoded).transpose(2, 0, 1))
    return np.stack(images)


def cache(samples, quality, deadline, *, crop_size=CROP):
    print(f"Decoding {len(samples)} images, JPEG Q{quality}", flush=True)
    views = np.empty((len(samples), 5, 3, crop_size, crop_size), dtype=np.uint8)
    sizes = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        decode = partial(prepare_image, quality=quality, crop_size=crop_size)
        for start in range(0, len(samples), 32):
            check_time(deadline)
            block = executor.map(decode, samples[start:start + 32])
            for index, (size, pixels) in enumerate(block, start=start):
                sizes.append(size)
                views[index] = pixels
    return torch.from_numpy(views), sizes


def check_time(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("The nightly time budget elapsed")


def sample_rows(samples, sizes, train_end, cohort=None):
    def ordered(start, stop, label):
        valid = [i for i in range(start, stop) if samples[i].label == label]
        return sorted(valid, key=lambda i: sha256(f"cohort-probe:{samples[i].image_id}".encode()).digest())

    real_train = ordered(0, train_end, 0)
    fake_train = ordered(0, train_end, 1)
    real_val = ordered(train_end, len(samples), 0)
    fake_val = ordered(train_end, len(samples), 1)
    if cohort is not None:
        fake_train = [i for i in fake_train if sizes[i] != (cohort, cohort)]
        fake_val = [i for i in fake_val if sizes[i] == (cohort, cohort)]
        real_val = real_val[:len(fake_val)]
    count = len(fake_train) // BATCH * BATCH
    if not count or not fake_val or len(real_val) != len(fake_val) or len(real_train) < count:
        raise ValueError(f"Insufficient samples for cohort {cohort}")
    return torch.tensor(real_train[:count] + fake_train[:count]), torch.tensor(real_val + fake_val)


@torch.inference_mode()
def scores(models, data, rows, device, deadline):
    for model in models:
        model.eval()
    result = []
    for ids in rows.split(BATCH):
        check_time(deadline)
        inputs = data[ids].to(device)
        results = [model(inputs).float().softmax(dim=1)[:, 1] for model in models]
        result.extend(torch.stack(results).mean(dim=0).cpu().tolist())
    return np.asarray(result)


def evaluate(models, data, targets, rows, device, deadline):
    values = scores(models, data, rows, device, deadline)
    return binary_metrics(targets[rows].tolist(), values > 0.5, values)


def within_class_alignment(embeddings, groups):
    vectors = F.normalize(embeddings.float(), dim=1)
    membership = F.one_hot(groups, num_classes=8).float()
    counts = membership.sum(dim=0)
    totals = membership.T @ vectors
    means = totals / counts.clamp(min=1)[:, None]
    centers = totals.reshape(2, 4, -1).sum(dim=1) / counts.reshape(2, 4).sum(dim=1).clamp(min=1)[:, None]
    error = (means - centers.repeat_interleave(4, dim=0)).square().sum(dim=1)
    present = counts > 1
    return (error * present).sum() / present.sum().clamp(min=1)


def fit(output, variant, seed, epochs, data, labels, rows, device, deadline, *,
        sampling_weights=None, paired_data=None, model_factory=PatchCNN, content_pairs=None,
        snapshot_epochs=(), domain_groups=None, subtype_targets=None, example_weights=None, objective=None,
        paired_objective=None):
    if paired_objective is not None and paired_data is None:
        raise ValueError("A paired objective needs paired training images")
    seed_all(seed, device)
    model = model_factory(variant).to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    accelerator = device.type in ("xpu", "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=accelerator, init_scale=4096)
    order = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        running = torch.zeros((), device=device)
        if content_pairs is not None:
            permutation = torch.randperm(len(content_pairs), generator=order)
            batch_rows = content_pairs[permutation].flatten()
        elif sampling_weights is None:
            permutation = torch.randperm(len(rows), generator=order)
            batch_rows = rows[permutation]
        else:
            permutation = torch.multinomial(sampling_weights, len(rows), replacement=True, generator=order)
            batch_rows = rows[permutation]
        for ids in batch_rows.split(BATCH):
            check_time(deadline)
            images, targets = data[ids].to(device), labels[ids].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=accelerator):
                if paired_objective is not None:
                    loss = paired_objective(model, images, paired_data[ids].to(device), targets, ids)
                elif subtype_targets is not None:
                    subtype_scores = model.subtype_scores(images)
                    logits = model.binary_logits(subtype_scores)
                elif domain_groups is None:
                    logits = model(images)
                else:
                    embeddings = model.patch_embeddings(images)
                    logits = model.classifier(embeddings).reshape(len(images), images.shape[1], 2).mean(dim=1)
                if paired_objective is None:
                    if objective is not None:
                        loss = objective(logits, targets, ids)
                    elif example_weights is None:
                        loss = F.cross_entropy(logits, targets)
                    else:
                        weights = example_weights[ids].to(device)
                        loss = (F.cross_entropy(logits, targets, reduction="none") * weights).mean()
                    if subtype_targets is not None:
                        loss = loss + 0.35 * model.subtype_loss(subtype_scores, targets, subtype_targets[ids].to(device))
                    if domain_groups is not None:
                        per_image = embeddings.reshape(len(images), images.shape[1], -1).mean(dim=1)
                        loss = loss + 0.5 * within_class_alignment(per_image, domain_groups[ids].to(device))
                    if content_pairs is not None:
                        evidence = (logits[:, 1].float() - logits[:, 0].float()).reshape(-1, 2)
                        loss = loss + 0.4 * F.softplus(0.5 - (evidence[:, 1] - evidence[:, 0])).mean()
                    if paired_data is not None:
                        alternate = model(paired_data[ids].to(device))
                        loss = 0.5 * (loss + F.cross_entropy(alternate, targets))
                        difference = logits.float().softmax(dim=1) - alternate.float().softmax(dim=1)
                        loss = loss + 0.5 * difference.square().mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.detach() * len(ids)
        average = running.item() / len(rows)
        if not np.isfinite(average):
            raise ValueError("Non-finite training loss")
        history.append({"epoch": epoch, "loss": average, "seconds": time.monotonic() - started})
        checkpoint = {"variant": variant, "seed": seed, "epoch": epoch, "weights": model.state_dict()}
        save_checkpoint(output / "last.pt", checkpoint)
        if epoch in snapshot_epochs:
            save_checkpoint(output / f"epoch-{epoch}.pt", checkpoint)
        write_json(output / "epochs.json", history)
        print(f"{output.parent.name}/{output.name} epoch {epoch}/{epochs} loss={average:.4f}", flush=True)
    return model


def nightly(output, report, device, deadline):
    def record(stage, **fields):
        report.update(stage=stage, updated=datetime.now().isoformat(timespec="seconds"), **fields)
        write_json(output / "status.json", report)

    args = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    split, identity = labeled_splits(args, {"train_per_class": 3200, "val_per_class": 800,
                                            "short_side": CROP, "crop_size": CROP})
    samples = split["train"] + split["val"]
    train_end = len(split["train"])
    labels = torch.tensor([item.label for item in samples])
    record("decode_labeled", split_sha256=identity["split_sha256"])
    data, sizes = cache(samples, 75, deadline)
    screening = {}
    record("screening", screening=screening)
    for variant in VARIANTS:
        per_cohort = {}
        screening[variant] = per_cohort
        for cohort in SCREEN_COHORTS:
            train_rows, val_rows = sample_rows(samples, sizes, train_end, cohort)
            runs = []
            per_cohort[str(cohort)] = runs
            for seed in SEEDS:
                root = output / f"screen_{cohort}_{variant}" / f"s{seed}"
                model = fit(root, variant, seed, SCREEN_EPOCHS, data, labels, train_rows, device, deadline)
                metric = evaluate([model], data, labels, val_rows, device, deadline)
                runs.append({"seed": seed, **metric})
                print(f"Screen {variant} {cohort} s{seed}: {metric['accuracy']:.3f}", flush=True)
                record("screening")
                del model

    ranked = []
    for variant, outcomes in screening.items():
        means = [float(np.mean([run["accuracy"] for run in runs])) for runs in outcomes.values()]
        print(f"Screen means {variant}: {means}", flush=True)
        if min(means) >= SCREEN_MIN and float(np.mean(means)) >= SCREEN_MEAN:
            ranked.append((min(means), float(np.mean(means)), variant))
    if not ranked:
        record("stopped_at_screen", reason="No direction passed both labeled screening gates")
        return

    winner = max(ranked)[2]
    confirmed = {}
    record("long_validation", winner=winner, long_validation=confirmed)
    for cohort in (*SCREEN_COHORTS, CONFIRM_COHORT):
        train_rows, val_rows = sample_rows(samples, sizes, train_end, cohort)
        models = [fit(output / f"long_{cohort}_{winner}" / f"s{seed}", winner, seed,
                      LONG_EPOCHS, data, labels, train_rows, device, deadline) for seed in SEEDS]
        metrics = {"q75": evaluate(models, data, labels, val_rows, device, deadline)}
        if cohort == CONFIRM_COHORT:
            harder, _ = cache([samples[i] for i in val_rows.tolist()], 90, deadline)
            metrics["q90"] = evaluate(models, harder, labels[val_rows], torch.arange(len(val_rows)), device, deadline)
        confirmed[str(cohort)] = metrics
        print(f"Long validation {cohort}: {metrics}", flush=True)
        record("long_validation")
        if any(result["accuracy"] < CONFIRM_MIN for result in metrics.values()):
            record("stopped_at_long_validation", reason=f"Cohort {cohort} fell below {CONFIRM_MIN:.0%}")
            return

    record("full_training")
    train_rows, val_rows = sample_rows(samples, sizes, train_end)
    models = [fit(output / f"full_{winner}" / f"s{seed}", winner, seed,
                  LONG_EPOCHS, data, labels, train_rows, device, deadline) for seed in SEEDS]
    metrics = evaluate(models, data, labels, val_rows, device, deadline)
    if metrics["accuracy"] < CONFIRM_MIN:
        record("stopped_at_full_validation", full_validation=metrics)
        return

    record("exporting", full_validation=metrics)
    del data
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    test_data, _ = cache(test, 75, deadline)
    predictions = scores(models, test_data, torch.arange(len(test)), device, deadline) > 0.5
    write_csv(output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(label)) for sample, label in zip(test, predictions, strict=True)))
    record("ready_for_kaggle", submission="submission.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minutes", type=int, default=90)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    args = parser.parse_args()
    if args.minutes < 1:
        parser.error("--minutes must be positive")
    deadline = time.monotonic() + args.minutes * 60
    report = {"stage": "starting", "pid": os.getpid(), "device": args.device, "minutes": args.minutes,
              "started": datetime.now().isoformat(timespec="seconds"), "screen_cohorts": SCREEN_COHORTS,
              "seeds": SEEDS, "screen_epochs": SCREEN_EPOCHS, "long_epochs": LONG_EPOCHS,
              "screen_min": SCREEN_MIN, "screen_mean": SCREEN_MEAN,
              "confirmation_cohort": CONFIRM_COHORT, "confirmation_min": CONFIRM_MIN}
    with run_lock(args.output):
        write_json(args.output / "status.json", report)
        awake = os.name == "nt" and ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            print(f"Night run PID {os.getpid()}, idle-sleep prevention requested: {bool(awake)}", flush=True)
            nightly(args.output, report, device_for(args.device, 6), deadline)
        except TimeoutError as exc:
            write_json(args.output / "status.json", {**report, "stage": "time_budget", "reason": str(exc)})
            print(str(exc), flush=True)
        except Exception as exc:
            write_json(args.output / "status.json", {**report, "stage": "failed", "reason": str(exc)})
            raise
        finally:
            if awake:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()

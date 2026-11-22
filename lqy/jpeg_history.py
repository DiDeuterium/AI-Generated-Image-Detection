"""Match the initial JPEG history of generated and real training images."""

import argparse
import io
import random
import time
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, JpegImagePlugin

from detection.data import read_test
from detection.metrics import binary_metrics
from detection.training import write_csv, write_json

from .cnn import device_for, run_lock
from .group_patch import ARCHITECTURE, restore_ensemble
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, cache, check_time, fit, image_patches, sample_rows, scores
from .robust import labeled_splits
from .spectrum_patch import accuracy_interval


SCREEN = (512, 1024, 256)
CONFIRM = 128
QUALITY = 75


def jpeg_profile(sample):
    with Image.open(sample.path) as image:
        if image.format != "JPEG":
            raise ValueError(f"Real training image is not a JPEG: {sample.image_id}")
        sampling = JpegImagePlugin.get_sampling(image)
        tables = image.quantization
        if sampling not in (0, 1, 2) or not {0, 1}.issubset(tables):
            return None
        return sampling, tuple(tables[0]), tuple(tables[1])


def match_image(job):
    sample, profile = job
    sampling, luminance, chrominance = profile
    with Image.open(sample.path) as original:
        rgb = original.convert("RGB")
    with io.BytesIO() as encoded:
        rgb.save(encoded, format="JPEG", quality=-1, subsampling=sampling,
                 qtables=[list(luminance), list(chrominance)])
        encoded.seek(0)
        with Image.open(encoded) as restored:
            return image_patches(restored.convert("RGB"), QUALITY)


class TrainingViews:
    def __init__(self, original, replacements, fake_indices):
        self.original = original
        self.replacements = replacements
        self.position = torch.full((len(original),), -1, dtype=torch.int64)
        self.position[fake_indices] = torch.arange(len(fake_indices))

    def __getitem__(self, indices):
        output = self.original[indices]
        positions = self.position[indices]
        included = positions >= 0
        output[included] = self.replacements[positions[included]]
        return output


def training_views(original, samples, profiles, rows, deadline):
    real = [i for i in rows.tolist() if samples[i].label == 0]
    fake = torch.tensor([i for i in rows.tolist() if samples[i].label == 1])
    observed = [profiles[i] for i in real if profiles[i] is not None]
    if not observed:
        raise ValueError("No compatible JPEG profiles among selected real training images")
    random.Random(75).shuffle(observed)
    entries = fake.tolist()
    jobs = [(samples[i], observed[j % len(observed)]) for j, i in enumerate(entries)]
    patches = np.empty((len(fake), 5, 3, CROP, CROP), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(jobs), 32):
            check_time(deadline)
            for i, view in enumerate(executor.map(match_image, jobs[start:start + 32]), start=start):
                patches[i] = view
    return TrainingViews(original, torch.from_numpy(patches), fake), len(observed)


def run(args, device, deadline):
    split_args = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    split, identity = labeled_splits(split_args, {"train_per_class": 3200, "val_per_class": 800,
                                                  "short_side": CROP, "crop_size": CROP})
    samples = split["train"] + split["val"]
    train_end = len(split["train"])
    labels = torch.tensor([sample.label for sample in samples])
    report = {"epochs": SCREEN_EPOCHS, "seeds": SEEDS, "jpeg_quality_at_inference": QUALITY,
              "split_sha256": identity["split_sha256"], "results": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("prepare_labeled")
    original, sizes = cache(samples, QUALITY, deadline)
    donor_rows = [i for i in range(train_end) if samples[i].label == 0]
    with ThreadPoolExecutor(max_workers=4) as executor:
        profiles = dict(zip(donor_rows, executor.map(jpeg_profile, (samples[i] for i in donor_rows)), strict=True))
    real_validation = sorted((i for i in range(train_end, len(samples)) if samples[i].label == 0),
                             key=lambda i: sha256(f"threshold-audit:{samples[i].image_id}".encode()).digest())
    offset = 0

    def train_pair(folder, data, rows):
        return [fit(folder / f"s{seed}", ARCHITECTURE, seed, SCREEN_EPOCHS, data, labels, rows,
                    device, deadline) for seed in SEEDS]

    def matched_pair(folder, rows):
        print(f"{folder.name}: matching JPEG history from its training partition", flush=True)
        matched, donor_count = training_views(original, samples, profiles, rows, deadline)
        return train_pair(folder, matched, rows), donor_count

    def fold(cohort):
        nonlocal offset
        train_rows, group = sample_rows(samples, sizes, train_end, cohort)
        fake = group[len(group) // 2:].tolist()
        rows = torch.tensor(real_validation[offset:offset + len(fake)] + fake)
        if len(rows) != 2 * len(fake):
            raise ValueError("Insufficient disjoint real validation samples")
        offset += len(fake)
        candidate, donor_count = matched_pair(args.output / f"matched_{cohort}", train_rows)
        if cohort == CONFIRM:
            reference = train_pair(args.output / "reference_128", original, train_rows)
        else:
            folder = (args.group_reference / "confirm_256_reference" if cohort == 256
                      else args.night_reference / f"screen_{cohort}_{ARCHITECTURE}")
            reference = restore_ensemble([folder / f"s{seed}" / "last.pt" for seed in SEEDS], device)
        truth = labels[rows].numpy()
        first = scores(reference, original, rows, device, deadline)
        second = scores(candidate, original, rows, device, deadline)
        base = binary_metrics(truth, first > 0.5, first)
        result = binary_metrics(truth, second > 0.5, second)
        print(f"{cohort}: accuracy {base['accuracy']:.3%} -> {result['accuracy']:.3%}; "
              f"AUROC {base['auroc']:.4f} -> {result['auroc']:.4f}", flush=True)
        return {"reference": base, "matched": result, "donor_count": donor_count,
                "delta": {key: result[key] - base[key] for key in ("accuracy", "auroc")},
                "paired_accuracy": accuracy_interval(truth, first, second)}

    for cohort in SCREEN:
        record(f"screen_{cohort}")
        report["results"][str(cohort)] = fold(cohort)
    wins = sum(all(change > 0 for change in item["delta"].values()) for item in report["results"].values())
    report["groups_improved_on_both_metrics"] = wins
    if wins >= 2:
        record("confirm_128")
        report["confirmation"] = fold(CONFIRM)

    record("full_training")
    train_rows, _ = sample_rows(samples, sizes, train_end)
    models, donor_count = matched_pair(args.output / "full_matched", train_rows)
    report["train_count"] = len(train_rows)
    report["full_training_donors"] = donor_count
    del original

    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"),
                     Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    views, _ = cache(test, QUALITY, deadline)
    prediction = scores(models, views, torch.arange(len(test)), device, deadline) > 0.5
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(label)) for sample, label in zip(test, prediction, strict=True)))
    record("complete")
    print(f"Wrote {len(test)} predictions", flush=True)


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

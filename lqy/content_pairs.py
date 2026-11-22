"""Screen training changes for a scratch native-patch detector."""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from detection.data import read_test
from detection.training import write_csv, write_json

from .cnn import device_for, run_lock
from .deep_fusion import CONFIRM, MIN_CONFIRM_GAIN, MIN_SCREEN_GAIN, SCREEN, bootstrap_mean_gain, combined_reference, comparison
from .group_patch import ARCHITECTURE
from .night_train import CROP, SEEDS, cache as inference_cache, check_time, fit, image_patches, sample_rows, scores
from .robust import labeled_splits


QUALITY = 75
SCREEN_EPOCHS = 12
FULL_EPOCHS = 20


def resample_patch(crop):
    # A 7/8 scale and a fractional JPEG-block shift spread the original block
    # boundaries before the model sees the deterministic Q75 recompression.
    compact = crop.resize((112, 112), Image.Resampling.LANCZOS)
    return compact.transform((CROP, CROP), Image.Transform.EXTENT, (2.5, 3.5, 110, 110), Image.Resampling.BICUBIC)


def prepare(sample, method):
    with Image.open(sample.path) as source:
        size = source.size
        rgb = source.convert("RGB")
    auxiliary = None
    if method == "content-pairs":
        auxiliary = np.asarray(rgb.resize((8, 8), Image.Resampling.BOX), dtype=np.float32).flatten() / 255
    if method == "resampling-consistency":
        auxiliary = image_patches(rgb, QUALITY, transform=resample_patch)
    return size, image_patches(rgb, QUALITY), auxiliary


def cache(samples, method, deadline):
    print(f"Preparing {method} data from {len(samples)} labeled images", flush=True)
    packed = np.empty((len(samples), 5, 3, CROP, CROP), dtype=np.uint8)
    auxiliary = None
    if method == "content-pairs":
        auxiliary = np.empty((len(samples), 8 * 8 * 3), dtype=np.float32)
    if method == "resampling-consistency":
        auxiliary = np.empty_like(packed)
    sizes = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        decode = partial(prepare, method=method)
        for start in range(0, len(samples), 32):
            check_time(deadline)
            for index, (size, views, extra) in enumerate(executor.map(decode, samples[start:start + 32]), start=start):
                sizes.append(size)
                packed[index] = views
                if auxiliary is not None:
                    auxiliary[index] = extra
    return torch.from_numpy(packed), sizes, torch.from_numpy(auxiliary) if auxiliary is not None else None


def pair_images(descriptors, labels, rows):
    truth = labels[rows]
    real, fake = rows[truth == 0], rows[truth == 1]
    if not len(real) or len(real) != len(fake):
        raise ValueError("Content pairing needs the same number of real and generated training images")
    scale = descriptors[rows].std(dim=0, correction=0).clamp(min=0.05)
    costs = torch.cdist(descriptors[fake] / scale, descriptors[real] / scale).numpy()
    rng = np.random.default_rng(17)
    random_match = rng.permutation(len(real))
    available = np.ones(len(real), dtype=bool)
    matches = np.empty(len(fake), dtype=np.int64)
    for index in rng.permutation(len(fake)):
        chosen = int(np.argmin(np.where(available, costs[index], np.inf)))
        matches[index] = chosen
        available[chosen] = False
    positions = np.arange(len(fake))
    summary = {"pairs": len(fake), "matched_distance": float(costs[positions, matches].mean()),
               "random_distance": float(costs[positions, random_match].mean())}
    return torch.stack((real[torch.from_numpy(matches)], fake), dim=1), summary


def training_groups(sizes, labels, rows):
    short_sides = np.asarray([min(sizes[i]) for i in rows.tolist()])
    fake = labels[rows].numpy() == 1
    quartiles = np.quantile(short_sides[~fake], (0.25, 0.5, 0.75))
    codes = np.searchsorted(quartiles, short_sides, side="right")
    codes[fake] = 4 + np.searchsorted((192, 384, 768), short_sides[fake], side="right")
    groups = torch.zeros_like(labels)
    groups[rows] = torch.from_numpy(codes)
    return groups, {"real_short_edge_quartiles": quartiles.tolist(),
                    "training_group_counts": np.bincount(codes, minlength=8).tolist()}


def run(args, device, deadline):
    data = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    split, identity = labeled_splits(data, {"train_per_class": 3200, "val_per_class": 800,
                                           "short_side": CROP, "crop_size": CROP})
    samples = split["train"] + split["val"]
    n_train = len(split["train"])
    labels = torch.tensor([sample.label for sample in samples])
    screen_epochs, full_epochs = (6, 6) if args.training == "resampling-consistency" else (SCREEN_EPOCHS, FULL_EPOCHS)
    report = {"initialization": "random", "epochs": screen_epochs, "full_epochs": full_epochs,
              "training_method": args.training,
              "seeds": SEEDS, "split_sha256": identity["split_sha256"], "screen": {},
              "reference": "56.5% spatial/spectrum rule", "quality": QUALITY,
              "gate": {"screen_mean_accuracy_gain": MIN_SCREEN_GAIN, "confirmation_gain": MIN_CONFIRM_GAIN,
                       "positive_auroc_and_accuracy_in_each_screen_cohort": True,
                       "positive_screen_bootstrap_lower_bound": True}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("prepare_labeled")
    views, sizes, auxiliary = cache(samples, args.training, deadline)
    real = sorted((i for i in range(n_train, len(samples)) if samples[i].label == 0),
                  key=lambda i: sha256(f"threshold-audit:{samples[i].image_id}".encode()).digest())
    offset = 0
    paired = []

    def train(folder, rows, epochs):
        if args.training == "group-alignment":
            groups, details = training_groups(sizes, labels, rows)
            settings = {"domain_groups": groups}
            print(f"{folder.name}: group counts {details['training_group_counts']}", flush=True)
        elif args.training == "content-pairs":
            pairs, details = pair_images(auxiliary, labels, rows)
            settings = {"content_pairs": pairs}
            print(f"{folder.name}: matched distance {details['matched_distance']:.3f}, "
                  f"random {details['random_distance']:.3f}", flush=True)
        else:
            settings = {"paired_data": auxiliary}
            details = {"paired_training_images": len(rows), "resample_pixels": 112}
        models = [fit(folder / f"s{seed}", ARCHITECTURE, seed, epochs, views, labels, rows, device, deadline,
                      **settings) for seed in SEEDS]
        return models, details

    def fold(cohort, epochs):
        nonlocal offset
        training, group = sample_rows(samples, sizes, n_train, cohort)
        fake = group[len(group) // 2:].tolist()
        rows = torch.tensor(real[offset:offset + len(fake)] + fake)
        if len(rows) != 2 * len(fake):
            raise ValueError("Insufficient disjoint real validation samples")
        offset += len(fake)
        models, distances = train(args.output / f"cohort_{cohort}", training, epochs)
        candidate = scores(models, views, rows, device, deadline)
        prior = combined_reference(args, cohort, views, labels, training, rows, device, deadline)
        truth = labels[rows].numpy()
        result = comparison(truth, prior, candidate)
        key = args.training.replace("-", "_")
        result[key] = result.pop("deep_fusion")
        detail_key = {"content-pairs": "matching", "group-alignment": "alignment_groups",
                      "resampling-consistency": "paired_augmentation"}[args.training]
        result[detail_key] = distances
        if cohort in SCREEN:
            paired.append((truth, prior, candidate))
        print(f"Cohort {cohort}: accuracy {result['reference']['accuracy']:.3%} -> "
              f"{result[key]['accuracy']:.3%}; AUROC {result['reference']['auroc']:.4f} -> "
              f"{result[key]['auroc']:.4f}", flush=True)
        return result

    for cohort in SCREEN:
        record(f"screen_{cohort}")
        report["screen"][str(cohort)] = fold(cohort, screen_epochs)
    grouped = bootstrap_mean_gain(paired)
    report["screen_paired_summary"] = grouped
    improved = all(all(change > 0 for change in item["delta"].values()) for item in report["screen"].values())
    if not improved or grouped["mean_gain"] < MIN_SCREEN_GAIN or grouped["bootstrap_95_percent"][0] <= 0:
        record("stopped_at_screen")
        print(f"Submission gate failed at screening: {grouped}", flush=True)
        return

    record("confirm_128")
    confirmation = fold(CONFIRM, full_epochs)
    report["confirmation"] = confirmation
    if confirmation["delta"]["accuracy"] < MIN_CONFIRM_GAIN or confirmation["delta"]["auroc"] <= 0:
        record("stopped_at_confirmation")
        print("Submission gate failed on the independent cohort", flush=True)
        return

    record("full_training")
    rows, _ = sample_rows(samples, sizes, n_train)
    models, report["full_matching"] = train(args.output / "full", rows, full_epochs)
    report["train_count"] = len(rows)
    del views, auxiliary
    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"),
                     Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    packed, _ = inference_cache(test, QUALITY, deadline)
    predicted = scores(models, packed, torch.arange(len(test)), device, deadline) > 0.5
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(label)) for sample, label in zip(test, predicted, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--training", choices=("content-pairs", "group-alignment", "resampling-consistency"),
                        default="content-pairs")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

"""Train an auxiliary pixel detector using labeled resampling transformations."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from detection.data import jpeg_roundtrip
from detection.metrics import binary_metrics
from detection.training import write_json

from .cnn import device_for, run_lock
from .codec_canonical import transform
from .codec_patch import labeled_data
from .group_patch import ARCHITECTURE
from .night_train import CROP, SCREEN_EPOCHS, check_time, fit, image_patches, scores


def unchanged(image):
    return image


def resize_training_image(image, assignment):
    factor = 0.58 + 0.86 * assignment[0] / 255
    size = tuple(max(1, round(side * factor)) for side in image.size)
    kernel = (Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS)[assignment[1] % 3]
    return image.resize(size, kernel)


def compression(image, assignment, profile):
    if profile == "random":
        codec = ("native", "jpeg", "jpeg", "webp")[assignment[2] % 4]
        if codec == "jpeg":
            return jpeg_roundtrip(image, quality=69 + assignment[3] % 27, subsampling=assignment[4] % 3)
        return transform(image, codec)
    return transform(image, profile)


def data_pair(item, profile, crops):
    with Image.open(item.path) as opened:
        rgb = opened.convert("RGB")
    assignment = sha256(f"resize-training:{item.image_id}".encode()).digest()
    resized = resize_training_image(rgb, assignment) if profile == "random" else transform(rgb, "resize")
    raw, affected = (compression(view, assignment, profile) for view in (rgb, resized))
    return tuple(image_patches(view, 75, encoder=unchanged)[crops] for view in (raw, affected))


def cache_pairs(items, profile, deadline, *, training=False):
    crops = [0, 4] if training else list(range(5))
    data = np.empty((2 * len(items), len(crops), 3, CROP, CROP), dtype=np.uint8)
    print(f"Preparing paired {profile} resampling labels for {len(items)} images", flush=True)
    with ThreadPoolExecutor(max_workers=3) as workers:
        for start in range(0, len(items), 32):
            check_time(deadline)
            batch = workers.map(lambda item: data_pair(item, profile, crops), items[start:start + 32])
            for index, (raw, affected) in enumerate(batch, start=start):
                data[index], data[len(items) + index] = raw, affected
    return torch.from_numpy(data)


def balanced_items(data, amount, offset=0):
    items = []
    for label in (0, 1):
        group = sorted((x for x in data if x.label == label),
                       key=lambda x: sha256(f"resize-probe-fold:{x.image_id}".encode()).digest())
        if len(group) < amount + offset:
            raise ValueError("Not enough labeled images to keep resampling calibration separate")
        items.extend(group[offset:offset + amount])
    return items


def resize_labels(count):
    return torch.arange(2, dtype=torch.int64).repeat_interleave(count)


def run(args, device, deadline):
    split, identity = labeled_data()
    training = balanced_items(split["train"], args.per_class)
    anchor = balanced_items(split["train"], 128, args.per_class)
    validation = balanced_items(split["val"], 240)
    report = {"split_sha256": identity["split_sha256"], "training_source_count": len(training),
              "calibration_source_count": len(anchor), "validation_source_count": len(validation), "profiles": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("training")
    packed = cache_pairs(training, "random", deadline, training=True)
    model = fit(args.output / "seed17", ARCHITECTURE, 17, SCREEN_EPOCHS, packed,
                resize_labels(len(training)), torch.arange(len(packed)), device, deadline)
    del packed
    anchor_packed = cache_pairs(anchor, "random", deadline)
    p = scores([model], anchor_packed, torch.arange(len(anchor_packed)), device, deadline)
    threshold = float(np.quantile(p[len(anchor):], 0.05))
    report["native_route_threshold"] = threshold
    del anchor_packed

    for profile in ("native", "jpeg76", "webp"):
        record(f"validate_{profile}")
        packed = cache_pairs(validation, profile, deadline)
        p = scores([model], packed, torch.arange(len(packed)), device, deadline)
        truth = resize_labels(len(validation)).tolist()
        measured = binary_metrics(truth, p >= threshold, p)
        report["profiles"][profile] = measured
        print(f"{profile}: AUROC {measured['auroc']:.3f}, original accepted {measured['recall_real']:.2%}, "
              f"rescaled accepted {1 - measured['recall_ai']:.2%}", flush=True)
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="cpu")
    parser.add_argument("--per-class", type=int, default=640)
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 4), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

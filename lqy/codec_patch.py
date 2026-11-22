"""Intervene on JPEG tables and train a scratch detector to agree across encodes."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
import io
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from detection.training import write_json

from .cnn import device_for, run_lock
from .deep_fusion import SCREEN, combined_reference, comparison
from .group_patch import ARCHITECTURE, restore_ensemble
from .jpeg_history import jpeg_profile
from .night_train import CROP, SEEDS, SCREEN_EPOCHS, cache, check_time, fit, image_patches, scores
from .offline_search import gate
from .phase_patch import run as run_candidate
from .robust import labeled_splits
from .spectrum_patch import QUALITY, accuracy_interval


RULES = {"codec_consistency": (ARCHITECTURE, 1.0), "codec_consistency_reference_mean": (ARCHITECTURE, 0.5)}


def labeled_data():
    args = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    return labeled_splits(args, {"train_per_class": 3200, "val_per_class": 800,
                                 "short_side": CROP, "crop_size": CROP})


def tables_by_frequency(samples):
    with ThreadPoolExecutor(max_workers=4) as workers:
        profiles = workers.map(jpeg_profile, (item for item in samples if item.label == 0))
        counts = Counter(profile[1:] for profile in profiles if profile is not None)
    common = counts.most_common(2)
    if len(common) != 2:
        raise ValueError("Two distinct training JPEG profiles are required")
    return tuple(table for table, _ in common), [
        {"n": count, "luminance_mean": float(np.mean(table[0])), "chroma_mean": float(np.mean(table[1]))}
        for table, count in common
    ]


def encode(crop, table):
    with io.BytesIO() as buffer:
        crop.save(buffer, format="JPEG", quality=-1, subsampling=2, qtables=[list(values) for values in table])
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


def prepare(item, *, crop_size=CROP):
    sample, table = item
    with Image.open(sample.path) as image:
        rgb = image.convert("RGB")
    return image_patches(rgb, QUALITY, encoder=partial(encode, table=table), crop_size=crop_size)


def cache_encodes(samples, tables, deadline, *, crop_size=CROP):
    pixels = np.empty((len(samples), 5, 3, crop_size, crop_size), dtype=np.uint8)
    jobs = list(zip(samples, tables, strict=True))
    with ThreadPoolExecutor(max_workers=4) as workers:
        decode = partial(prepare, crop_size=crop_size)
        for start in range(0, len(jobs), 32):
            check_time(deadline)
            for i, result in enumerate(workers.map(decode, jobs[start:start + 32]), start=start):
                pixels[i] = result
    return torch.from_numpy(pixels)


def shift_report(truth, p, q):
    flips = (p > 0.5) != (q > 0.5)
    return {"mean_absolute_score_shift": float(np.abs(p - q).mean()), "flip_rate": float(flips.mean()),
            "flip_rate_real": float(flips[truth == 0].mean()), "flip_rate_ai": float(flips[truth == 1].mean()),
            "accuracy": float(((q > 0.5) == truth).mean()), "paired_vs_reference": accuracy_interval(truth, p, q)}


def audit(args, device, deadline, split, identity, profiles):
    validation = split["val"]
    labels = torch.tensor([sample.label for sample in validation])
    reference, sizes = cache(validation, QUALITY, deadline)
    ordering = sorted(range(len(validation)), key=lambda i: sha256(f"threshold-audit:{validation[i].image_id}".encode()).digest())
    real = [i for i in ordering if validation[i].label == 0]
    offset = 0
    result = {}
    vote_weights = {str(weight): (weight, weight) for weight in (0.25, 0.5, 0.75, 1.0)}
    vote_weights.update({"change_spatial": (1.0, 0.0), "change_spectrum": (0.0, 1.0),
                         "mix_spatial": (0.5, 0.0), "mix_spectrum": (0.0, 0.5)})
    paired = {name: [] for name in vote_weights}
    for cohort in SCREEN:
        fake = [i for i in ordering if validation[i].label == 1 and sizes[i] == (cohort, cohort)]
        ids = real[offset:offset + len(fake)] + fake
        offset += len(fake)
        positions = torch.arange(len(ids))
        subset = [validation[i] for i in ids]
        truth = labels[ids].numpy()

        def score(images):
            return np.stack(combined_reference(args, cohort, images, labels[ids], None, positions, device, deadline,
                                               return_components=True))

        native = reference[ids]
        first = cache_encodes(subset, [profiles[0]] * len(ids), deadline)
        original_components = score(native)
        alternate_components = score(first)
        baseline = original_components.mean(axis=0)
        alternatives = [alternate_components.mean(axis=0),
                        score(cache_encodes(subset, [profiles[1]] * len(ids), deadline)).mean(axis=0)]
        votes = {}
        for name, weight in vote_weights.items():
            weights = np.asarray(weight)[:, None]
            prediction = ((1 - weights) * original_components + weights * alternate_components).mean(axis=0)
            votes[name] = comparison(truth, baseline, prediction)
            paired[name].append((truth, baseline, prediction))
        result[str(cohort)] = {"n": len(ids), "accuracy_q75": float(((baseline > 0.5) == truth).mean()),
                               "dominant": shift_report(truth, baseline, alternatives[0]),
                               "second": shift_report(truth, baseline, alternatives[1]),
                               "dominant_to_second": shift_report(truth, alternatives[0], alternatives[1]),
                               "fixed_vote": votes}
        result[str(cohort)]["component_shifts"] = {
            name: shift_report(truth, original, changed)
            for name, original, changed in zip(("spatial", "spectrum"), original_components, alternate_components, strict=True)
        }
        if args.compare_checkpoint is not None:
            folder = args.compare_checkpoint / f"group-{cohort}" / ARCHITECTURE
            models = restore_ensemble([folder / f"s{seed}" / "last.pt" for seed in SEEDS], device)
            q75 = scores(models, native, positions, device, deadline)
            changed = scores(models, first, positions, device, deadline)
            measured = shift_report(truth, q75, changed)
            result[str(cohort)]["trained_consistency"] = measured
            print(f"Cohort {cohort}: trained consistency model flips {measured['flip_rate']:.2%}", flush=True)
        print(f"Cohort {cohort}: flip Q75 to dominant {result[str(cohort)]['dominant']['flip_rate']:.2%}, "
              f"dominant to second {result[str(cohort)]['dominant_to_second']['flip_rate']:.2%}", flush=True)
    votes = {name: gate({cohort: data["fixed_vote"][name] for cohort, data in result.items()}, comparisons)
             for name, comparisons in paired.items()}
    write_json(args.output / "intervention.json", {"split_sha256": identity["split_sha256"], "cohorts": result,
                                                   "fixed_vote_gates": votes})


def train_variant(folder, variant, views, labels, _sizes, rows, device, deadline, *, paired):
    return [fit(folder / f"s{seed}", variant, seed, SCREEN_EPOCHS, views, labels, rows, device, deadline,
                paired_data=paired) for seed in SEEDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--mode", choices=("audit", "train"), required=True)
    parser.add_argument("--compare-checkpoint", type=Path)
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        device = device_for(args.device, 6)
        deadline = time.monotonic() + 3600
        split, identity = labeled_data()
        profiles, description = tables_by_frequency(split["train"])
        write_json(args.output / "profiles.json", description)
        print(f"Train-derived JPEG tables: {description}", flush=True)
        if args.mode == "audit":
            audit(args, device, deadline, split, identity, profiles)
        else:
            samples = split["train"]
            assignments = [profiles[sha256(f"codec-treatment:{sample.image_id}".encode()).digest()[0] % 2]
                           for sample in samples]
            print(f"Encoding paired views for {len(samples)} training images", flush=True)
            paired = cache_encodes(samples, assignments, deadline)
            run_candidate(args, device, deadline, rules=RULES, train_model=partial(train_variant, paired=paired))


if __name__ == "__main__":
    main()

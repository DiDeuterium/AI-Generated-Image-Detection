"""Pair geometry-compatible sources without repeating training images."""
import csv
import math
from collections import Counter

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .target_profile import process


class PairedTargetSampler(Sampler):
    """Give both labels the same profile and similar resampling histories."""
    def __init__(self, rows, profiles, metadata, seed, fraction=.5, max_scale=1.5):
        if not 0 < fraction <= 1 or not .95 <= max_scale:
            raise ValueError("Invalid fraction or maximum scale")
        with metadata.open(newline="") as file:
            sizes = {row["image_id"]: (int(row["width"]), int(row["height"]))
                     for row in csv.DictReader(file) if row["split"] == "train"}
        self.n = len(rows)
        self.seed = seed
        self.epoch = 0
        labels = np.asarray([row["label"] for row in rows])
        source_sizes = np.asarray([sizes[row["image_id"]] for row in rows])
        counts = np.bincount(labels, minlength=2)
        if not counts[0] or counts[0] != counts[1]:
            raise ValueError("Expected equally populated real and fake training groups")
        self.requested_pairs = round(counts[0] * fraction)
        self.candidates = []
        weights = []
        for profile in profiles:
            scales = (np.asarray(profile[:2]) / source_sizes).max(axis=1)
            buckets = np.digitize(scales, (.75, 1., 1.25))
            bins = []
            for bucket in range(4):
                eligible = np.flatnonzero((scales <= max_scale) & (buckets == bucket))
                bins.append(tuple(eligible[labels[eligible] == label] for label in (0, 1)))
            self.candidates.append(bins)
            weights.append(profile[-1])
        self.support = np.asarray([sum(min(len(left), len(right)) for left, right in bins)
                                   for bins in self.candidates])
        self.profile_widths = [profile[0] for profile in profiles]
        weights = np.asarray(weights, dtype=float)
        weights[self.support < 20] = 0
        if not weights.any():
            raise ValueError("No target profiles with adequate paired support")
        self.weights = weights / weights.sum()

    def __len__(self):
        return self.n

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        requested = rng.choice(len(self.weights), self.requested_pairs, p=self.weights)
        requested = sorted(requested, key=lambda profile: self.support[profile])
        assigned = {}
        used = np.zeros(self.n, dtype=bool)
        cohorts = Counter()
        for profile in requested:
            available = [(left[~used[left]], right[~used[right]])
                         for left, right in self.candidates[profile]]
            capacity = np.asarray([min(len(left), len(right)) for left, right in available])
            if not capacity.any():
                continue
            bucket = rng.choice(len(available), p=capacity / capacity.sum())
            chosen = [int(rng.choice(indices)) for indices in available[bucket]]
            center = tuple(float(value) for value in rng.random(2))
            for source in chosen:
                assigned[source] = (int(profile), center)
                used[source] = True
            cohorts["width500" if self.profile_widths[profile] == 500 else "other"] += 1
        print(f"[paired] epoch={self.epoch} requested_pairs={self.requested_pairs} "
              f"matched_pairs={len(assigned) // 2} width500_pairs={cohorts['width500']} "
              f"other_pairs={cohorts['other']}", flush=True)
        return iter((int(source), *assigned.get(source, (None, None)))
                    for source in rng.permutation(self.n))


class PairedTrainingImages(Dataset):
    def __init__(self, rows, profiles, native_transform, fallback_transform, *, other_blur_radius=0.):
        if not math.isfinite(other_blur_radius) or other_blur_radius < 0:
            raise ValueError("Blur radius must be finite and nonnegative")
        self.other_blur_radius = other_blur_radius
        self.rows = rows
        self.profiles = profiles
        self.native_transform = native_transform
        self.fallback_transform = fallback_transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, assignment):
        source, target, center = assignment
        row = self.rows[source]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
        if target is None:
            tensor = self.fallback_transform(image)
        else:
            matched = process(image, self.profiles[target], "both", center,
                              other_blur_radius=self.other_blur_radius)
            tensor = self.native_transform(matched)
        return tensor, row["label"], source

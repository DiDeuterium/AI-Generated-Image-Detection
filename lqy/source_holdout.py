"""Compare scratch CNNs on held-out real JPEG families and generated subtypes."""

import argparse
from collections import Counter
from functools import partial
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detection.training import write_json

from .cnn import device_for, read_json, run_lock
from .codec_anchor import cutoff, prepare_data, recenter
from .codec_invariant import training_views
from .codec_route import NEAR_WEIGHT, WEIGHT, cache_routed
from .codec_scale import HYBRID, QUANTILE, HybridResponseCNN, contrast_components, restore_candidate
from .codec_support import tagged
from .deep_fusion import comparison
from .jpeg_history import jpeg_profile
from .night_train import BATCH, SCREEN_EPOCHS, SEEDS, fit, scores
from .phase_patch import full_reference


VARIANTS = ("constrained_rgb", "constrained_residual")
CONTRASTIVE = "cross_source_contrastive"
ANCHORS = 192
PROFILES = ((None, 2), (82, 0))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels),
                                  nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels))

    def forward(self, inputs):
        return F.relu(inputs + self.conv(inputs))


def block(in_channels, out_channels, kernel, stride):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride=stride, padding=kernel // 2, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU())


class SourceCNN(HybridResponseCNN):
    def __init__(self, variant):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown source-transfer variant: {variant}")
        super().__init__(HYBRID)
        self.rgb = variant == VARIANTS[0]
        kernels = torch.empty(9, 1, 5, 5)
        nn.init.normal_(kernels, std=0.10)
        self.high_pass = nn.Parameter(kernels)
        self.features = nn.Sequential(block(24 if self.rgb else 21, 24, 5, 4), ResidualBlock(24),
                                      block(24, 40, 3, 2), ResidualBlock(40),
                                      block(40, 64, 3, 2), ResidualBlock(64))

    def patch_features(self, views):
        rgb, residue, change, scale = contrast_components(views)
        kernels = self.high_pass - self.high_pass.mean(dim=(2, 3), keepdim=True)
        learned = F.conv2d(F.pad(rgb, (2, 2, 2, 2), mode="reflect"), kernels, groups=3)
        features = (6 * residue, 6 * change, (residue / scale).tanh(), (change / scale).tanh(), 6 * learned)
        return torch.cat((rgb, *features) if self.rgb else features, dim=1)


class ContrastiveCNN(HybridResponseCNN):
    def __init__(self, variant):
        if variant != CONTRASTIVE:
            raise ValueError(f"Unknown cross-source model: {variant}")
        super().__init__(HYBRID)
        self.projector = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 32))


def contrastive_loss(source, model, images, alternate, labels, ids):
    batch = len(ids)
    patches = model.patch_embeddings(torch.cat((images, alternate)))
    features = patches.reshape(2 * batch, images.shape[1], -1).mean(dim=1)
    classes = labels.repeat(2)
    logits = model.classifier(features)
    fit_loss = F.cross_entropy(logits, classes)
    first, second = logits.float().softmax(dim=1).chunk(2)
    consistency = (first - second).square().mean()

    embedding = F.normalize(model.projector(features).float(), dim=1)
    similarity = embedding @ embedding.T / 0.15
    domain = source[ids].to(labels.device).repeat(2)
    image_ids = torch.arange(batch, device=labels.device).repeat(2)
    distinct = ~torch.eye(2 * batch, dtype=torch.bool, device=labels.device)
    same_label = classes[:, None] == classes
    positive = same_label & distinct & ((domain[:, None] != domain) | (image_ids[:, None] == image_ids))
    active = positive | ~same_label
    normalizer = similarity.masked_fill(~active, -torch.inf).logsumexp(dim=1)
    average_positive = (similarity * positive).sum(dim=1) / positive.sum(dim=1)
    return fit_loss + 0.5 * consistency + 0.10 * (normalizer - average_positive).mean()


def source_folds(samples, labels, sizes, official_end):
    real_profiles = {i: jpeg_profile(item) for i, item in enumerate(samples) if item.label == 0}
    frequency = Counter(profile for i, profile in real_profiles.items() if i < official_end)
    dominant = frequency.most_common(1)[0][0]
    marked = {i for i, item in enumerate(samples) if item.label == 1 and sizes[i] == (256, 256) and tagged(item)}
    large = {i for i, item in enumerate(samples) if item.label == 1 and sizes[i] == (1024, 1024)}

    def sorted_ids(indices, group):
        return sorted(indices, key=lambda i: sha256(f"source-transfer:{group}:{samples[i].image_id}".encode()).digest())

    folds = {}
    for name, hold_fake, hold_dominant in (("marked_256", marked, False), ("native_1024", large, True)):
        hold_real = {i for i, profile in real_profiles.items() if (profile == dominant) == hold_dominant}
        real_train = sorted_ids([i for i in real_profiles if i < official_end and i not in hold_real], name)
        fake_train = sorted_ids([i for i in range(official_end) if labels[i] == 1 and i not in hold_fake], name)
        anchor = real_train[-ANCHORS:]
        count = min(len(real_train) - len(anchor), len(fake_train)) // BATCH * BATCH
        train = torch.tensor(real_train[:count] + fake_train[:count])
        val_real = sorted_ids([i for i in hold_real if i >= official_end], name)
        val_fake = sorted_ids([i for i in hold_fake if i >= official_end], name)
        count_val = min(len(val_real), len(val_fake))
        if count == 0 or count_val == 0 or len(anchor) != ANCHORS:
            raise ValueError(f"Insufficient source-specific training or validation images for {name}")
        val = torch.tensor(val_real[:count_val] + val_fake[:count_val])
        folds[name] = train, torch.tensor(anchor), val
    return folds


class ReleasedReference:
    def __init__(self, args, device, identity):
        self.args, self.device = args, device
        roots = ((args.narrow_reference, WEIGHT), (args.near_reference, NEAR_WEIGHT))
        self.parts = []
        for root, weight in roots:
            state = read_json(root / "summary.json")
            if state["split_sha256"] != identity:
                raise ValueError("Released checkpoint has a different labeled split")
            models = restore_candidate(root / "full" / HYBRID, HYBRID, HybridResponseCNN, device)
            self.parts.append((models, state["threshold"], weight))

    def score(self, native, packed, guard, rows, deadline):
        baseline = full_reference(self.args, native, rows, self.device, deadline)
        result = baseline.copy()
        for mask, (model, threshold, weight) in zip((~guard, guard), self.parts, strict=True):
            if mask.any():
                prediction = recenter(scores(model, packed, rows[torch.from_numpy(mask)], self.device, deadline), threshold)
                result[mask] = (1 - weight) * baseline[mask] + weight * prediction
        return result


def source_gain(folds):
    rng = np.random.default_rng(17)
    estimates = np.zeros(4000)
    means = []
    for truth, views in folds:
        changes = np.stack([((value > 0.5) == truth).astype(float) - ((baseline > 0.5) == truth)
                            for baseline, value in views]).mean(axis=0)
        means.append(float(changes.mean()))
        for label in (0, 1):
            group = changes[truth == label]
            sample = rng.integers(len(group), size=(len(estimates), len(group)))
            estimates += group[sample].sum(axis=1) / (len(folds) * len(truth))
    return {"mean_gain": float(np.mean(means)), "paired_95_percent": np.quantile(estimates, (0.025, 0.975)).tolist()}


def run(args, device, deadline):
    samples, labels, sizes, _, _, full, table, identity = prepare_data()
    folds = source_folds(samples, labels, sizes, len(full))
    reference = ReleasedReference(args, device, identity)
    matched = read_json(args.matched_reference / "summary.json") if args.matched_reference is not None else None
    if matched is not None and (matched["stage"] != "complete" or matched["split_sha256"] != identity):
        raise ValueError("Cross-source contrast needs the finished same-split controls")
    variants = (CONTRASTIVE,) if matched is not None else VARIANTS
    report = {"split_sha256": identity, "profiles": PROFILES, "epochs": SCREEN_EPOCHS, "seeds": SEEDS,
              "variants": variants, "cohorts": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    pairs = {name: [] for name in variants}
    for name, (train, anchor, val) in folds.items():
        record(f"training_{name}")
        all_rows = torch.cat((train, anchor))
        items = [samples[i] for i in all_rows.tolist()]
        print(f"{name}: {len(train)} training images, {len(anchor)} calibration photographs, {len(val)} validation images", flush=True)
        native, compressed, _ = training_views(items, table, deadline)
        local_labels = labels[all_rows]
        training_rows = torch.arange(len(train))
        anchor_rows = torch.arange(len(train), len(items))
        checkpoints, thresholds = {}, {}
        if matched is not None:
            checkpoints[HYBRID] = restore_candidate(args.matched_reference / name / HYBRID, HYBRID, HybridResponseCNN, device)
            thresholds[HYBRID] = matched["cohorts"][name]["thresholds"][HYBRID]
        new_training = (HYBRID, *variants) if matched is None else variants
        for variant in new_training:
            factory = HybridResponseCNN if variant == HYBRID else SourceCNN
            objective = None
            if variant == CONTRASTIVE:
                factory = ContrastiveCNN
                groups = [2 * int(np.searchsorted((192, 384, 768), min(sizes[i]), side="right"))
                          + int(samples[i].label == 1 and sizes[i] == (256, 256) and tagged(samples[i]))
                          for i in all_rows.tolist()]
                objective = partial(contrastive_loss, torch.tensor(groups))
            models = [fit(args.output / name / variant / f"s{seed}", variant, seed, SCREEN_EPOCHS, native,
                          local_labels, training_rows, device, deadline, model_factory=factory, paired_data=compressed,
                          paired_objective=objective)
                      for seed in SEEDS]
            calibration = np.concatenate((scores(models, native, anchor_rows, device, deadline),
                                          scores(models, compressed, anchor_rows, device, deadline)))
            checkpoints[variant] = models
            thresholds[variant] = cutoff(calibration, QUANTILE)
        del native, compressed

        truth = labels[val].numpy()
        items = [samples[i] for i in val.tolist()]
        validation_rows = torch.arange(len(val))
        cohort_report = {"train_count": len(train), "calibration_count": len(anchor), "validation_count": len(val),
                         "views": {}, "thresholds": thresholds}
        cohort_predictions = {variant: [] for variant in variants}
        for quality, subsampling in PROFILES:
            original, views, guard = cache_routed(items, quality, table, deadline, subsampling=subsampling)
            scores_by_variant = {variant: recenter(scores(models, views, validation_rows, device, deadline), thresholds[variant])
                                 for variant, models in checkpoints.items()}
            existing = reference.score(original, views, guard, validation_rows, deadline)
            control = scores_by_variant[HYBRID]
            view_report = {}
            for variant in variants:
                prediction = scores_by_variant[variant]
                equal_support = comparison(truth, control, prediction)
                against_release = comparison(truth, existing, prediction)
                cohort_predictions[variant].append((control, prediction))
                view_report[variant] = {"equal_support": equal_support, "against_61_version": against_release}
                print(f"{name} {quality} {variant}: matched {equal_support['reference']['accuracy']:.2%} -> "
                      f"{equal_support['deep_fusion']['accuracy']:.2%}; released {against_release['reference']['accuracy']:.2%}", flush=True)
            cohort_report["views"]["native" if quality is None else f"q{quality}"] = view_report
        report["cohorts"][name] = cohort_report
        for variant, values in cohort_predictions.items():
            pairs[variant].append((truth, values))
        record(f"done_{name}")

    report["aggregate"] = {variant: source_gain(results) for variant, results in pairs.items()}
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--matched-reference", type=Path, help="Reuse completed controls to test cross-source contrast")
    parser.add_argument("--narrow-reference", type=Path, default=Path("lqy/runs/codec-final/upstream-trained"))
    parser.add_argument("--near-reference", type=Path, default=Path("lqy/runs/codec-final/near-upstream"))
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 5400)


if __name__ == "__main__":
    main()

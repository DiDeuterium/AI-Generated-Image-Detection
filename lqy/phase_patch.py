"""Train scratch CNNs on neighboring-pixel differences and compare held-out groups."""

import argparse
import time
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from detection.data import read_test
from detection.training import write_csv, write_json

from .cnn import device_for, run_lock
from .deep_fusion import CONFIRM, MIN_CONFIRM_GAIN, SCREEN, combined_reference, comparison, spectral_views
from .group_patch import restore_ensemble
from .heldout_blend import restore as restore_reference
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, PatchCNN, cache, fit, sample_rows, scores
from .offline_search import gate
from .robust import labeled_splits
from .spectrum_patch import QUALITY, VARIANT as SPECTRUM_VARIANT, SpectrumPatchCNN


VARIANTS = ("npr", "polyphase")
RULES = {"npr": ("npr", 1.0), "polyphase": ("polyphase", 1.0),
         "npr_reference_mean": ("npr", 0.5), "polyphase_reference_mean": ("polyphase", 0.5)}


def cell_differences(pixels):
    a, b = pixels[..., ::2, ::2], pixels[..., ::2, 1::2]
    c, d = pixels[..., 1::2, ::2], pixels[..., 1::2, 1::2]
    return torch.cat((a - b + c - d, a + b - c - d, a - b - c + d), dim=1) * 0.5


class PhasePatchCNN(PatchCNN):
    def __init__(self, variant):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown phase variant: {variant}")
        super().__init__("residual")
        self.variant = variant
        if variant == "polyphase":
            self.features[0] = nn.Conv2d(36, 16, 3, padding=1, stride=2, bias=False)

    def patch_features(self, views):
        rgb = views.flatten(0, 1).float() / 255
        if self.variant == "npr":
            anchor = rgb[..., ::2, ::2].repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
            return (rgb - anchor) * 6

        native = cell_differences(rgb)
        coarse = cell_differences(F.avg_pool2d(rgb, 2))
        # Keep signed contrast and its magnitude at each scale.
        return torch.cat((F.avg_pool2d(native, 2), F.avg_pool2d(native.abs(), 2),
                          coarse, coarse.abs()), dim=1) * 6


def fit_variant(folder, variant, views, labels, _sizes, rows, device, deadline):
    return [fit(folder / f"s{seed}", variant, seed, SCREEN_EPOCHS, views, labels,
                rows, device, deadline, model_factory=PhasePatchCNN) for seed in SEEDS]


def full_reference(args, views, rows, device, deadline, *, spatial_only=False):
    spatial_folder = args.night_reference / "online-probe"
    spatial = restore_ensemble([spatial_folder / f"s{seed}" / "last.pt" for seed in SEEDS], device)
    spatial_scores = scores(spatial, views, rows, device, deadline)
    if spatial_only:
        return spatial_scores
    frequency = restore_reference(args.spectrum_reference / "online-probe", SpectrumPatchCNN, SPECTRUM_VARIANT, device)
    packed = spectral_views(views, rows, deadline)
    return (spatial_scores + scores(frequency, packed, torch.arange(len(rows)), device, deadline)) * 0.5


def run(args, device, deadline, *, rules=RULES, train_model=fit_variant, include_all_real=False, candidate_data=None,
        blend_with_spatial=False):
    data = SimpleNamespace(manifest=Path("splits/train_val.csv"), image_root=Path("dataset"))
    split, identity = labeled_splits(data, {"train_per_class": 3200, "val_per_class": 800,
                                           "short_side": CROP, "crop_size": CROP})
    samples = split["train"] + split["val"]
    end = len(split["train"])
    labels = torch.tensor([sample.label for sample in samples])
    real = sorted((i for i in range(end, len(samples)) if samples[i].label == 0),
                  key=lambda i: sha256(f"threshold-audit:{samples[i].image_id}".encode()).digest())
    report = {"initialization": "random", "split_sha256": identity["split_sha256"], "epochs": SCREEN_EPOCHS,
              "seeds": SEEDS, "quality": QUALITY, "rules": rules, "screen": {},
              "blend_reference": "spatial" if blend_with_spatial else "spatial_spectrum"}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("prepare_labeled")
    views, sizes = cache(samples, QUALITY, deadline)
    model_views = views if candidate_data is None else candidate_data(samples, views, deadline)
    offset = 0
    paired = {name: [] for name in rules}

    def fold(cohort, candidates):
        nonlocal offset
        train, group = sample_rows(samples, sizes, end, cohort)
        selected = train
        if include_all_real:
            all_real = torch.where(labels[:end] == 0)[0]
            selected = torch.cat((all_real, train[labels[train] == 1]))
        fake = group[len(group) // 2:].tolist()
        val = torch.tensor(real[offset:offset + len(fake)] + fake)
        if len(val) != 2 * len(fake):
            raise ValueError("Insufficient disjoint real validation images")
        offset += len(fake)
        reference = combined_reference(args, cohort, views, labels, train, val, device, deadline,
                                       return_components=blend_with_spatial)
        if blend_with_spatial:
            spatial, frequency = reference
            reference = (spatial + frequency) * 0.5
            blend_reference = spatial
        else:
            blend_reference = reference
        raw = {}
        for variant in dict.fromkeys(rules[name][0] for name in candidates):
            models = train_model(args.output / f"group-{cohort}" / variant, variant, model_views, labels, sizes, selected,
                                 device, deadline)
            raw[variant] = scores(models, model_views, val, device, deadline)
        truth = labels[val].numpy()
        report["confirmation" if cohort == CONFIRM else "screen"][str(cohort)] = results = {}
        for name in candidates:
            variant, weight = rules[name]
            predicted = weight * raw[variant] + (1 - weight) * blend_reference
            result = comparison(truth, reference, predicted)
            result["candidate"] = result.pop("deep_fusion")
            results[name] = result
            if cohort in SCREEN:
                paired[name].append((truth, reference, predicted))
            print(f"{cohort} {name}: accuracy {result['reference']['accuracy']:.3%} -> "
                  f"{result['candidate']['accuracy']:.3%}, AUROC {result['reference']['auroc']:.4f} -> "
                  f"{result['candidate']['auroc']:.4f}", flush=True)
        return results

    for cohort in SCREEN:
        record(f"screen_{cohort}")
        fold(cohort, rules)
        record(f"screen_{cohort}_complete")
    gates = {name: gate({group: entry[name] for group, entry in report["screen"].items()}, pairs)
             for name, pairs in paired.items()}
    report["gates"] = gates
    candidates = sorted((name for name, result in gates.items() if result["passes"]),
                        key=lambda name: gates[name]["mean_gain"], reverse=True)
    if not candidates:
        record("stopped_at_screen")
        return

    report["confirmation"] = {}
    record(f"confirm_{CONFIRM}")
    confirmation = fold(CONFIRM, candidates)
    passing = [name for name in candidates if confirmation[name]["delta"]["accuracy"] >= MIN_CONFIRM_GAIN
               and confirmation[name]["delta"]["auroc"] > 0]
    if not passing:
        record("stopped_at_confirmation")
        return

    selected = passing[0]
    variant, weight = rules[selected]
    report["selected"] = selected
    record("full_training")
    train, _ = sample_rows(samples, sizes, end)
    models = train_model(args.output / "full" / variant, variant, model_views, labels, sizes, train, device, deadline)
    del views, model_views
    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    views, _ = cache(test, QUALITY, deadline)
    rows = torch.arange(len(test))
    model_views = views if candidate_data is None else candidate_data(test, views, deadline)
    probability = scores(models, model_views, rows, device, deadline) * weight
    if weight < 1:
        probability += (1 - weight) * full_reference(args, views, rows, device, deadline,
                                                      spatial_only=blend_with_spatial)
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(score > 0.5)) for sample, score in zip(test, probability, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

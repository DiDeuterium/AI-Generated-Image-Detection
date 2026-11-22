"""Train a JPEG-response classifier whose features resist compression-domain clues."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from detection.data import jpeg_roundtrip, read_test
from detection.training import write_csv, write_json

from .cnn import device_for, read_json, run_lock
from .codec_anchor import cutoff, exclude_anchor, prepare_data, recenter
from .codec_final import profile_gain
from .codec_patch import encode
from .codec_response import PairedViews
from .codec_route import NEAR_WEIGHT, WEIGHT, cache_routed
from .codec_scale import HYBRID, QUANTILE, HybridResponseCNN, restore_candidate
from .deep_fusion import comparison
from .night_train import CROP, SCREEN_EPOCHS, SEEDS, check_time, fit, image_patches, scores
from .phase_patch import full_reference
from .spectrum_patch import QUALITY


VARIANT = "codec_invariant"
TRAIN_PROFILES = ((68, 0), (73, 2), (81, 0), (90, 2))
TRAIN_CROPS = (0, 1, 4)
SCREEN = ((None, 2), (73, 2), (88, 1), (69, 0), (79, 0))
CONFIRM = ((76, 1), (84, 0))
WEIGHTS = (0.5, 1.0)


class InvariantCNN(HybridResponseCNN):
    def __init__(self, variant):
        if variant != VARIANT:
            raise ValueError(f"Unknown compression invariant model {variant}")
        super().__init__(HYBRID)
        self.domain = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 32), nn.GELU(),
                                    nn.Linear(32, 2 * (len(TRAIN_PROFILES) + 1)))


def paired_loss(domains, model, original, alternate, labels, rows):
    batch = len(original)
    stacked = torch.cat((original, alternate))
    features = model.patch_embeddings(stacked).reshape(2 * batch, original.shape[1], -1).mean(dim=1)
    logits = model.classifier(features)
    original_logits, alternate_logits = logits.float().chunk(2)
    classification = F.cross_entropy(logits, labels.repeat(2))
    consistency = (original_logits.softmax(dim=1) - alternate_logits.softmax(dim=1)).square().mean()
    reversed_features = features.detach() + 0.25 * (features.detach() - features)
    domain_scores = model.domain(reversed_features).reshape(2 * batch, 2, -1)
    per_class = domain_scores[torch.arange(2 * batch, device=labels.device), labels.repeat(2)]
    transformed = domains[rows].to(labels.device)
    expected = torch.cat((torch.zeros_like(transformed), transformed))
    return classification + 0.5 * consistency + 0.35 * F.cross_entropy(per_class, expected)


def training_views(items, table, deadline):
    assigned = [sha256(f"invariant-profile:{item.image_id}".encode()).digest()[0] % len(TRAIN_PROFILES) for item in items]
    shape = (len(items), len(TRAIN_CROPS), 3, CROP, CROP)
    native = np.empty(shape, dtype=np.uint8)
    native_alt = np.empty_like(native)
    stress = np.empty_like(native)
    stress_alt = np.empty_like(native)
    indices = list(TRAIN_CROPS)
    encoder = partial(encode, table=table)

    def render(job):
        item, position = job
        with Image.open(item.path) as source:
            rgb = source.convert("RGB")
        quality, subsampling = TRAIN_PROFILES[position]
        altered = jpeg_roundtrip(rgb, quality=quality, subsampling=subsampling)
        return (image_patches(rgb, QUALITY)[indices], image_patches(rgb, QUALITY, encoder=encoder)[indices],
                image_patches(altered, QUALITY)[indices], image_patches(altered, QUALITY, encoder=encoder)[indices])

    jobs = list(zip(items, assigned, strict=True))
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(jobs), 32):
            check_time(deadline)
            for i, result in enumerate(workers.map(render, jobs[start:start + 32]), start=start):
                native[i], native_alt[i], stress[i], stress_alt[i] = result
            if start % 1600 == 0:
                print(f"Prepared {min(start + 32, len(jobs))}/{len(jobs)} labeled training-pool images", flush=True)
    return (PairedViews(torch.from_numpy(native), torch.from_numpy(native_alt)),
            PairedViews(torch.from_numpy(stress), torch.from_numpy(stress_alt)), torch.tensor(assigned) + 1)


def run(args, device, deadline):
    samples, labels, _, anchor, _, full, table, identity = prepare_data()
    if not torch.equal(torch.sort(full).values, torch.arange(len(full))):
        raise ValueError("Unexpected official training partition")
    train = exclude_anchor(full, labels, anchor)
    selected = torch.cat((train, anchor))
    report = {"split_sha256": identity, "train_count": len(train), "anchor_count": len(anchor),
              "training_profiles": TRAIN_PROFILES, "training_crops": TRAIN_CROPS, "screen": {}, "confirmation": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    record("training")
    native, stress, domains = training_views([samples[i] for i in selected.tolist()], table, deadline)
    training_labels = labels[selected]
    train_positions = torch.arange(len(train))
    candidate = [fit(args.output / "full" / VARIANT / f"s{seed}", VARIANT, seed, SCREEN_EPOCHS, native,
                     training_labels, train_positions, device, deadline, model_factory=InvariantCNN, paired_data=stress,
                     paired_objective=partial(paired_loss, domains)) for seed in SEEDS]
    calibration = torch.arange(len(train), len(selected))
    threshold = cutoff(np.concatenate((scores(candidate, native, calibration, device, deadline),
                                       scores(candidate, stress, calibration, device, deadline))), QUANTILE)
    report["threshold"] = threshold
    del native, stress

    incumbent = read_json(args.released_reference / "summary.json")
    narrow = read_json(args.narrow_reference / "summary.json")
    near = read_json(args.near_reference / "summary.json")
    if (incumbent["stage"] != "complete" or incumbent["blend_weight"] != WEIGHT or incumbent["near_weight"] != NEAR_WEIGHT
            or any(item["split_sha256"] != identity for item in (incumbent, narrow, near))):
        raise ValueError("The comparison checkpoints must match the released route and labeled split")
    narrow_models = restore_candidate(args.narrow_reference / "full" / HYBRID, HYBRID, HybridResponseCNN, device)
    near_models = restore_candidate(args.near_reference / "full" / HYBRID, HYBRID, HybridResponseCNN, device)
    truth = labels[len(full):].numpy()
    validation = samples[len(full):]

    def evaluate(items, quality, subsampling):
        raw, paired, guarded = cache_routed(items, quality, table, deadline, subsampling=subsampling)
        rows = torch.arange(len(items))
        reference = full_reference(args, raw, rows, device, deadline)
        current = reference.copy()
        for mask, models, state, weight in ((~guarded, narrow_models, narrow, WEIGHT),
                                            (guarded, near_models, near, NEAR_WEIGHT)):
            if mask.any():
                result = recenter(scores(models, paired, rows[torch.from_numpy(mask)], device, deadline), state["threshold"])
                current[mask] = (1 - weight) * reference[mask] + weight * result
        return current, recenter(scores(candidate, paired, rows, device, deadline), threshold)

    def verify(profiles, weights, key):
        paired = {weight: [] for weight in weights}
        for quality, subsampling in profiles:
            name = "native" if quality is None else f"q{quality}_{subsampling}"
            record(f"validate_{name}")
            before, new = evaluate(validation, quality, subsampling)
            report[key][name] = {}
            for weight in weights:
                candidate_scores = (1 - weight) * before + weight * new
                result = comparison(truth, before, candidate_scores)
                report[key][name][str(weight)] = result
                paired[weight].append((before, candidate_scores))
                print(f"{name} weight={weight}: {result['reference']['accuracy']:.3%} -> "
                      f"{result['deep_fusion']['accuracy']:.3%}", flush=True)
            record(f"validate_{name}_complete")
        return {weight: profile_gain(truth, values) for weight, values in paired.items()}

    screen = verify(SCREEN, WEIGHTS, "screen")
    report["screen_gain"] = screen
    passing = [weight for weight, gain in screen.items()
               if gain["mean_gain"] >= 0.02 and gain["paired_95_percent"][0] > 0
               and all(items[str(weight)]["delta"]["accuracy"] >= -0.015 for items in report["screen"].values())]
    if not passing:
        record("stopped_at_screen")
        return
    chosen = max(passing, key=lambda weight: screen[weight]["mean_gain"])
    report["selected_weight"] = chosen
    confirmation = verify(CONFIRM, (chosen,), "confirmation")[chosen]
    report["confirmation_gain"] = confirmation
    if (confirmation["paired_95_percent"][0] <= 0
            or any(items[str(chosen)]["delta"]["accuracy"] < -0.01 for items in report["confirmation"].values())):
        record("stopped_at_confirmation")
        return

    record("export")
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected 1,000 test images")
    before, after = evaluate(test, None, 2)
    values = (1 - chosen) * before + chosen * after
    write_csv(args.output / "submission.csv", ("image_id", "label"),
              ((sample.image_id, int(score > 0.5)) for sample, score in zip(test, values, strict=True)))
    record("complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--released-reference", type=Path, default=Path("lqy/runs/codec-final/near-specialist"))
    parser.add_argument("--narrow-reference", type=Path, default=Path("lqy/runs/codec-final/upstream-trained"))
    parser.add_argument("--near-reference", type=Path, default=Path("lqy/runs/codec-final/near-upstream"))
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 3600)


if __name__ == "__main__":
    main()

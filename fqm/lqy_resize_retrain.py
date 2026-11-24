"""Train the LQY resize expert from scratch and compare it with FQM's frozen ensemble."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
from pathlib import Path
import csv
import json
import time

import numpy as np
from PIL import Image
import torch

from detection.data import jpeg_roundtrip
from lqy.canonical_augmentation import CORRELATION_VARIANT, CorrelationCNN, canonical_views
from lqy.codec_anchor import recenter
from lqy.codec_canonical import CANONICAL, calibration_rules
from lqy.codec_patch import encode, labeled_data, tables_by_frequency
from lqy.codec_response import PairedViews
from lqy.night_train import SEEDS, SCREEN_EPOCHS, fit, image_patches, scores
from lqy.resize_specialist import near_resize
from lqy.spectrum_patch import QUALITY

from ssd.data import build_loader, build_transform, load_norm_stats, read_test_ids
from ssd.domain_shift import legacy_degrade
from ssd.engine import load_checkpoint, predict, resolve_amp
from ssd.target_profile import FixedTargetMatch, validation_groups
from train import build_model, load_config

BASE_NAMES = ("target_control_20261121", "target_matched_20261121")
BASE_VAL = Path("fqm/runs/ensemble_views_20261121/center")
ARCHIVE = Path("fqm/submissions/exp-7047-7048_ensemble.csv")
TEMPLATE = Path("ai-generated-image-detection-benchmark/sample_submission.csv")
TEST_DIR = Path("dataset/dataset/test")
CASES = ("orig", "both", "legacy_strong")
WEIGHTS = (0., .25, .5, .75, 1.)


def worst(metrics):
    return min(metrics[group] for group in ("width500", "other", "strong"))


def select_weight(results):
    baseline = results[0.]
    eligible = [weight for weight, result in results.items()
                if result["target"] >= baseline["target"] - .010000001]
    return max(eligible, key=lambda weight: (worst(results[weight]), results[weight]["target"], -weight))


def confirmation_passes(base, candidate):
    return (candidate["target"] >= base["target"] - .010000001
            and (worst(candidate) - worst(base) >= .009999999
                 or candidate["target"] - base["target"] >= .009999999))


def training_split(samples):
    per_class = [[row for row in samples if row.label == label] for label in (0, 1)]
    for group in per_class:
        group.sort(key=lambda sample: sha256(f"full-canonical-anchor:{sample.image_id}".encode()).digest())
        if len(group) != 3200:
            raise ValueError("Expected 3,200 training images per class")
    return [row for group in per_class for row in group[:-320]], [row for group in per_class for row in group[-320:]]


def image_views(image, table):
    canonical = jpeg_roundtrip(image, quality=CANONICAL, subsampling=2)
    primary = image_patches(canonical, QUALITY)
    alternate = image_patches(canonical, QUALITY, encoder=partial(encode, table=table))
    return primary, alternate


def prepare_views(row, table, transform):
    with Image.open(row["path"]) as opened:
        image = opened.convert("RGB")
    return image_views(transform(image, row["image_id"]), table)


def specialist_scores(models, rows, table, transform, device, deadline):
    prepare = partial(prepare_views, table=table, transform=transform)
    batches = []
    with ThreadPoolExecutor(max_workers=4) as workers:
        for start in range(0, len(rows), 64):
            primary, alternate = zip(*workers.map(prepare, rows[start:start + 64]), strict=True)
            data = PairedViews(torch.from_numpy(np.stack(primary)), torch.from_numpy(np.stack(alternate)))
            batches.append(scores(models, data, torch.arange(len(primary)), device, deadline))
    return np.concatenate(batches)


def identity(image, _):
    return image


def strong(image, image_id):
    return legacy_degrade(image, image_id, "strong")


def accuracy_summary(values, labels, cohorts, subset):
    target = (values["both"] > .5) == labels
    result = {
        "target": float(target[subset].mean()),
        "width500": float(target[subset & (cohorts == "width500")].mean()),
        "other": float(target[subset & (cohorts == "other")].mean()),
        "strong": float(((values["legacy_strong"][subset] > .5) == labels[subset]).mean()),
        "orig": float(((values["orig"][subset] > .5) == labels[subset]).mean()),
    }
    for label, name in ((0, "real"), (1, "fake")):
        result[f"target_{name}_recall"] = float(target[subset & (labels == label)].mean())
    return result


def blend(baseline, expert, weight):
    return {case: (1 - weight) * baseline[case] + weight * expert[case] for case in CASES}


def frozen_probs(rows, device):
    each = []
    for name in BASE_NAMES:
        config = load_config(Path("fqm/tune/configs") / f"{name}.json")
        data = config["data"]
        stats = load_norm_stats(Path(data["norm_stats"]))
        transform = build_transform(data["input_size"], data["resize_size"], stats["mean"],
                                    stats["std"], train=False, aug=data["aug"])
        loader = build_loader(rows, transform, 32, False, 2, 0)
        model = build_model(config["model"]).to(device)
        load_checkpoint(Path("fqm/runs/tune") / name / "best.pt", model, device)
        amp, dtype = resolve_amp(config["train"]["amp"], config["train"]["amp_dtype"], device)
        each.append(predict(model, loader, device, amp, dtype)[:, 1])
    return np.mean(each, axis=0)


def export(models, table, threshold, weight, device, deadline, output, csv_path):
    test = read_test_ids(TEMPLATE, TEST_DIR)
    ids = [row["image_id"] for row in test]
    if len(ids) != 1000 or len(set(ids)) != len(ids):
        raise ValueError("Invalid submission template")
    baseline = frozen_probs(test, device)
    with ARCHIVE.open(newline="") as file:
        archive = list(csv.DictReader(file))
    if ids != [row["image_id"] for row in archive]:
        raise ValueError("Archived submission order differs from the template")
    np.testing.assert_array_equal(baseline > .5, [int(row["label"]) for row in archive])
    expert = specialist_scores(models, test, table, identity, device, deadline)
    combined = (1 - weight) * baseline + weight * recenter(expert, threshold)
    np.savez(output / "test_scores.npz", ids=ids, baseline=baseline, expert=expert, combined=combined)
    with csv_path.open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image_id", "label"))
        writer.writerows(zip(ids, (combined > .5).astype(int), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 85 * 60
    split, details = labeled_data()
    train, anchors = training_split(split["train"])
    tables, profile_info = tables_by_frequency(split["train"])
    table = tables[0]
    first, second = canonical_views(train, table, deadline, paired=True, alter_both=True, acquisition=near_resize)
    labels = torch.tensor([row.label for row in train])
    models = [fit(output / f"s{seed}", CORRELATION_VARIANT, seed, SCREEN_EPOCHS, first, labels,
                  torch.arange(len(train)), device, deadline, model_factory=CorrelationCNN, paired_data=second)
              for seed in SEEDS]
    del first, second
    anchor_rows = [{"path": row.path, "image_id": row.image_id} for row in anchors]
    anchor = specialist_scores(models, anchor_rows, table, identity, device, deadline)
    threshold = calibration_rules(anchor, np.array([row.label for row in anchors]))["both_class"]
    val = [{"path": row.path, "image_id": row.image_id, "label": row.label} for row in split["val"]]
    by_id = {row["image_id"]: row for row in val}
    _, held = validation_groups(val)
    held_ids = {row["image_id"] for row in held}
    profiles = json.loads(Path("fqm/runs/target_profile_20261121/profiles.json").read_text())
    match = FixedTargetMatch(val, profiles, "both", seed=126)
    with np.load(BASE_VAL / f"{BASE_NAMES[0]}.npz") as a, np.load(BASE_VAL / f"{BASE_NAMES[1]}.npz") as b:
        for key in ("ids", "labels", "cohorts"):
            np.testing.assert_array_equal(a[key], b[key])
        ids, truth, cohorts = (a[key] for key in ("ids", "labels", "cohorts"))
        reference = {case: (a[case] + b[case]) * .5 for case in CASES}
    assert len(ids) == len(set(ids)) == 1600
    val = [by_id[key] for key in ids]
    np.testing.assert_array_equal(truth, [row["label"] for row in val])
    np.testing.assert_array_equal(cohorts, [match.cohort(key) for key in ids])
    holdout = np.array([key in held_ids for key in ids])
    assert holdout.sum() == 800
    expert = {}
    for case, transform in zip(CASES, (identity, match, strong), strict=True):
        print(f"Validating expert on {case}", flush=True)
        expert[case] = recenter(specialist_scores(models, val, table, transform, device, deadline), threshold)
    selection = {weight: accuracy_summary(blend(reference, expert, weight), truth, cohorts, ~holdout)
                 for weight in WEIGHTS}
    weight = select_weight(selection)
    confirmation = {name: accuracy_summary(blend(reference, expert, value), truth, cohorts, holdout)
                    for name, value in (("baseline", 0.), ("expert", 1.), ("selected", weight))}
    accepted = weight > 0 and confirmation_passes(confirmation["baseline"], confirmation["selected"])
    report = {"source_commit": "b23beec", "split_sha256": details["split_sha256"],
              "seeds": SEEDS, "epochs_per_seed": SCREEN_EPOCHS, "train_count": len(train),
              "calibration_count": len(anchors), "quantization": profile_info[0],
              "expert_threshold": threshold, "selection": selection, "selected_weight": weight,
              "confirmation": confirmation, "passed_offline_gate": bool(accepted)}
    np.savez(output / "validation_scores.npz", ids=ids, labels=truth, cohorts=cohorts, holdout=holdout,
             **{f"reference_{k}": v for k, v in reference.items()}, **{f"expert_{k}": v for k, v in expert.items()})
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if accepted:
        export(models, table, threshold, weight, device, deadline, output, args.csv)
        print(f"CSV verified against template: {args.csv}", flush=True)


if __name__ == "__main__":
    main()

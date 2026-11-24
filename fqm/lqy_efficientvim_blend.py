"""Check whether a separately trained EfficientViM ensemble complements the 0.646 model."""
import argparse
import csv
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from lqy.codec_anchor import prepare_data, recenter
from lqy.codec_canonical import transform, validation_halves
from lqy.codec_final import profile_gain
from lqy.night_train import SEEDS

from lqy_compound_blend import COMPOUND, REPLAY, blend
from lqy_replay import NEW, PROFILES, branch, metrics, read_json, write_csv
from lqy_resize_retrain import (BASE_NAMES, TEMPLATE, TEST_DIR, identity, read_test_ids,
                                specialist_scores)
from lqy_specialist_gate import SCREEN
from ssd.data import ImageDataset, build_transform, load_norm_stats
from ssd.engine import load_checkpoint, predict, resolve_amp
from train import build_model, load_config

RUNS = Path("fqm/runs")
SELECTED = RUNS / "lqy_gate_20261122/selection_scores.npz"
BLEND_DIR = RUNS / "lqy_compound_blend_20261122"
CURRENT_CSV = Path("fqm/submissions/exp-lqy-compound-blend-20261122.csv")
RULES = [(mode, weight) for mode in ("average", "uncertain") for weight in (.10, .20, .30)]


def fuse(primary, secondary, rule):
    mode, weight = rule
    merged = (1 - weight) * primary + weight * secondary
    if mode == "uncertain":
        return np.where(np.abs(primary - .5) <= .20, merged, primary)
    return merged


def perturb(profile, image, _key):
    return transform(image, profile)


def evaluate(truth, panels, rule):
    result, pairs = {}, []
    for profile, (primary, secondary) in panels.items():
        adjusted = fuse(primary, secondary, rule)
        before = metrics(truth, primary)
        after = metrics(truth, adjusted)
        result[profile] = {"baseline": before, "candidate": after,
                           "delta": after["accuracy"] - before["accuracy"]}
        pairs.append((primary, adjusted))
    return result, profile_gain(truth, pairs)


class EfficientViM:
    def __init__(self, device):
        self.device = device
        self.models = []
        for name in BASE_NAMES:
            config = load_config(Path("fqm/tune/configs") / f"{name}.json")
            data = config["data"]
            stats = load_norm_stats(Path(data["norm_stats"]))
            preprocessing = build_transform(data["input_size"], data["resize_size"], stats["mean"],
                                            stats["std"], train=False, aug=data["aug"])
            model = build_model(config["model"]).to(device)
            load_checkpoint(RUNS / "tune" / name / "best.pt", model, device)
            amp, dtype = resolve_amp(config["train"]["amp"], config["train"]["amp_dtype"], device)
            self.models.append((model, preprocessing, amp, dtype))

    def score(self, rows, processing):
        probabilities = []
        for model, preprocessing, amp, dtype in self.models:
            loader = DataLoader(ImageDataset(rows, preprocessing, processing), batch_size=32,
                                num_workers=2, pin_memory=self.device.type == "cuda")
            probabilities.append(predict(model, loader, self.device, amp, dtype)[:, 1])
        return np.mean(probabilities, axis=0)


def rows_for(samples, indexes):
    return [{"path": samples[i].path, "image_id": samples[i].image_id, "label": samples[i].label}
            for i in indexes]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    if args.candidate.exists():
        raise FileExistsError(args.candidate)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 20 * 60
    samples, labels, sizes, _, _, train, table, split = prepare_data()
    for directory in (REPLAY, COMPOUND):
        assert split == read_json(directory / "summary.json")["split_sha256"]
    selection, holdout = validation_halves(samples, labels, sizes, len(train))
    efficient = EfficientViM(device)
    experts = [branch([run / f"s{seed}" / "last.pt" for seed in SEEDS], device)
               for run in (NEW, COMPOUND)]
    thresholds = [read_json(NEW / "summary.json")["expert_threshold"],
                  read_json(COMPOUND / "summary.json")["threshold"]]

    rows = rows_for(samples, selection)
    truth = labels[selection].numpy()
    selection_panels = {}
    with np.load(SELECTED) as cache:
        np.testing.assert_array_equal(cache["ids"], [row["image_id"] for row in rows])
        np.testing.assert_array_equal(cache["labels"], truth)
        for profile in SCREEN:
            processing = partial(perturb, profile)
            print(f"Selection: {profile}", flush=True)
            score = [recenter(specialist_scores(nets, rows, table, processing, device, deadline), cutoff)
                     for nets, cutoff in zip(experts, thresholds, strict=True)]
            primary = .35 * cache[f"{profile}_base"] + .45 * score[0] + .20 * score[1]
            selection_panels[profile] = primary, efficient.score(rows, processing)
    del experts
    trials = []
    for rule in RULES:
        result, gain = evaluate(truth, selection_panels, rule)
        trials.append({"rule": rule, "gain": gain, "profiles": result})
    eligible = [trial for trial in trials if trial["gain"]["mean_gain"] > 0.
                and min(p["delta"] for p in trial["profiles"].values()) >= -.010000001]
    report = {"baseline_online": .646, "selection": trials, "exported_candidate": False}
    summary_file = args.output / "summary.json"
    summary_file.write_text(json.dumps(report, indent=2) + "\n")
    if not eligible:
        print("No rule improved the selection split without a large profile loss.", flush=True)
        return
    best = max(eligible, key=lambda x: x["gain"]["mean_gain"])
    rule = best["rule"]

    rows = rows_for(samples, holdout)
    truth = labels[holdout].numpy()
    confirmation_panels = {}
    with np.load(REPLAY / "validation_scores.npz") as ref, np.load(COMPOUND / "validation_scores.npz") as compound:
        for source in (ref, compound):
            np.testing.assert_array_equal(source["ids"], [row["image_id"] for row in rows])
            np.testing.assert_array_equal(source["labels"], truth)
        for profile in PROFILES:
            print(f"Confirmation: {profile}", flush=True)
            _, baseline = blend(ref[f"{profile}_base"], ref[f"{profile}_replacement"], compound[profile])
            confirmation_panels[profile] = baseline, efficient.score(rows, partial(perturb, profile))
    final, gain = evaluate(truth, confirmation_panels, rule)
    report.update(rule=rule, confirmation=final, gain=gain)
    summary_file.write_text(json.dumps(report, indent=2) + "\n")
    np.savez(args.output / "validation_scores.npz", labels=truth,
             ids=[row["image_id"] for row in rows],
             **{f"{profile}_{key}": scores[i] for profile, scores in confirmation_panels.items()
                for i, key in enumerate(("base", "efficientvim"))})
    print(json.dumps({"rule": rule, "gain": gain, "deltas": {p: m["delta"] for p, m in final.items()}}), flush=True)

    if (gain["mean_gain"] <= 0. or final["alternate_resize_jpeg"]["delta"] < -.010000001
            or min(profile["delta"] for profile in final.values()) < -.015000001):
        return
    test = read_test_ids(TEMPLATE, TEST_DIR)
    ids = [row["image_id"] for row in test]
    with np.load(BLEND_DIR / "test_scores.npz") as cache, CURRENT_CSV.open(newline="") as file:
        np.testing.assert_array_equal(cache["ids"], ids)
        past = list(csv.DictReader(file))
        reference = cache["candidate"]
    assert len(ids) == len(set(ids)) == 1000
    np.testing.assert_array_equal([row["image_id"] for row in past], ids)
    np.testing.assert_array_equal([int(row["label"]) for row in past], reference > .5)
    candidate = fuse(reference, efficient.score(test, identity), rule)
    if not np.isfinite(candidate).all():
        raise ValueError("Non-finite score")
    changed = int(((candidate > .5) != (reference > .5)).sum())
    if changed:
        write_csv(args.candidate, ids, candidate)
        report.update(exported_candidate=True, changed_test_predictions=changed)
        summary_file.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Exported {args.candidate}: {changed} changed labels", flush=True)


if __name__ == "__main__":
    main()

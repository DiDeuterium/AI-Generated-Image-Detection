"""Select a conservative specialist override and confirm it on separate images."""
import argparse
import csv
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch

from lqy.canonical_augmentation import CORRELATION_VARIANT, CorrelationCNN
from lqy.codec_anchor import prepare_data, recenter
from lqy.codec_canonical import compare_views, validation_halves
from lqy.codec_final import profile_gain
from lqy.codec_scale import restore_candidate
from lqy.night_train import scores
from lqy.source_holdout import ReleasedReference
from lqy_replay import BUNDLE, PROFILES, read_json, write_csv
from lqy_resize_retrain import TEMPLATE

REPLAY = Path("fqm/runs/lqy_replay_20261122")
SCREEN = ("native", "jpeg83", "resize", "webp")


def override(base, released, expert_floor, minimum_base, weight):
    expert = (released - .75 * base) / .25
    combined = (1 - weight) * base + weight * expert
    allowed = (expert >= expert_floor) & (base >= minimum_base)
    return np.where(allowed, np.maximum(released, combined), released)


def evaluate(raw, labels, profiles, rule):
    results, pairs = {}, []
    for profile in profiles:
        base, released = raw[profile]
        candidate = override(base, released, *rule)
        old, new = released > .5, candidate > .5
        delta = ((new == labels).astype(int) - (old == labels))
        def summarize(predicted):
            return {"accuracy": float((predicted == labels).mean()),
                    "real_recall": float((~predicted[labels == 0]).mean()),
                    "fake_recall": float(predicted[labels == 1].mean())}
        results[profile] = {"old": summarize(old), "new": summarize(new),
                            "gain": float(delta.mean()), "changed": int((new != old).sum())}
        pairs.append((released, candidate))
    return results, pairs


def predict(items, profile, table, reference, specialist, threshold, device, deadline):
    raw, codec, resized, routing = compare_views(items, profile, table, deadline)
    positions = torch.arange(len(items))
    base = reference.score(raw, codec, routing, positions, deadline)
    expert = recenter(scores(specialist, resized, positions, device, deadline), threshold)
    return base, .75 * base + .25 * expert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 15 * 60
    manifest = read_json(BUNDLE / "manifest.json")
    releases = BUNDLE / "lqy/runs"
    samples, labels, sizes, _, _, full, table, split = prepare_data()
    assert split == read_json(REPLAY / "summary.json")["split_sha256"]
    selection, heldout = validation_halves(samples, labels, sizes, len(full))
    reference = ReleasedReference(SimpleNamespace(
        narrow_reference=releases / "codec-final/upstream-trained",
        near_reference=releases / "codec-final/near-upstream",
        night_reference=releases / "night-train/2026-11-21",
        spectrum_reference=releases / "spectrum-patch/round-01",
    ), device, split)
    specialist = restore_candidate(releases / "resize-specialist/round-01/full" / CORRELATION_VARIANT,
                                   CORRELATION_VARIANT, CorrelationCNN, device)
    threshold = manifest["thresholds"]["resize_specialist"]
    items = [samples[index] for index in selection]
    screen = {profile: predict(items, profile, table, reference, specialist, threshold, device, deadline)
              for profile in SCREEN}
    np.savez(args.output / "selection_scores.npz", ids=[row.image_id for row in items],
             labels=labels[selection].numpy(), **{f"{case}_{key}": values[i] for case, values in screen.items()
                                                   for i, key in enumerate(("base", "original"))})
    rules = product((.55, .65, .75, .85), (0., .15, .25, .35), (.5, .75, 1.))
    viable = []
    for rule in rules:
        result, _ = evaluate(screen, labels[selection].numpy(), SCREEN, rule)
        gains = [case["gain"] for case in result.values()]
        if min(gains) >= -.010000001 and result["resize"]["gain"] >= .02:
            viable.append((float(np.mean(gains)), result["resize"]["gain"], rule, result))
    report = {"split_sha256": split, "eligible_rules": len(viable), "candidate_exported": False}
    if viable:
        _, _, rule, result = max(viable, key=lambda item: item[:2])
        with np.load(REPLAY / "validation_scores.npz") as cache:
            np.testing.assert_array_equal(cache["ids"], [samples[index].image_id for index in heldout])
            np.testing.assert_array_equal(cache["labels"], labels[heldout].numpy())
            confirmation = {case: (cache[f"{case}_base"], cache[f"{case}_original"]) for case in PROFILES}
        final, pairs = evaluate(confirmation, labels[heldout].numpy(), PROFILES, rule)
        gain = profile_gain(labels[heldout].numpy(), pairs)
        report.update(rule=rule, selection=result, confirmation=final, gain=gain)
        if (gain["mean_gain"] >= .01 and gain["paired_95_percent"][0] > 0
                and final["alternate_resize_jpeg"]["gain"] >= .02
                and all(row["gain"] >= -.010000001 for row in final.values())):
            with np.load(REPLAY / "test_scores.npz") as cache, TEMPLATE.open(newline="") as file:
                ids = [row["image_id"] for row in csv.DictReader(file)]
                assert len(ids) == len(set(ids)) == 1000
                np.testing.assert_array_equal(cache["ids"], ids)
                previous = cache["original"]
                predictions = override(cache["base"], previous, *rule)
            changed = int(((previous > .5) != (predictions > .5)).sum())
            if changed:
                write_csv(args.candidate, ids, predictions)
                report.update(candidate_exported=True, changed_test_predictions=changed)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

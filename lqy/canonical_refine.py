"""Verify asymmetric correction of missed generated images."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from detection.data import read_test
from detection.training import write_csv, write_json

from .canonical_augmentation import CORRELATION_VARIANT, CorrelationCNN
from .cnn import device_for, read_json, run_lock
from .codec_anchor import prepare_data, recenter
from .codec_canonical import CONFIRM, SCREEN, WEIGHTS, compare_views, validation_halves
from .codec_final import profile_gain
from .codec_scale import restore_candidate
from .deep_fusion import comparison
from .night_train import scores
from .overnight_training import TARGET_CORRECT, assess_export
from .source_holdout import ReleasedReference


def recover_generated(before, after, weight):
    """Promote a missed detection when the new evidence crosses the boundary."""
    return np.maximum(before, (1 - weight) * before + weight * after)


def blend(before, after, weight):
    return (1 - weight) * before + weight * after


def agreeing_blend(before, specialist, partner, weight):
    default = blend(before, specialist, 0.25)
    same_prediction = (specialist > 0.5) == (partner > 0.5)
    strong = np.minimum(np.abs(specialist - 0.5), np.abs(partner - 0.5)) >= 0.10
    consensus = (specialist + partner) * 0.5
    return np.where(same_prediction & strong, blend(default, consensus, weight), default)


def run(args, device, deadline):
    samples, labels, sizes, _, _, train, table, identity = prepare_data()
    selection, confirmation = validation_halves(samples, labels, sizes, len(train))
    state = read_json(args.correlation / "summary.json")
    if state["split_sha256"] != identity or state["variant"] != CORRELATION_VARIANT or "threshold" not in state:
        raise ValueError("A completed correlation training run is required")
    models = restore_candidate(args.correlation / "full" / CORRELATION_VARIANT,
                               CORRELATION_VARIANT, CorrelationCNN, device)
    released = ReleasedReference(args, device, identity)
    combine = blend if args.verify_specialist else recover_generated
    partner_models, partner_threshold = None, None
    if args.agreement_reference is not None:
        partner_state = read_json(args.agreement_reference / "summary.json")
        if partner_state["split_sha256"] != identity or partner_state["variant"] != CORRELATION_VARIANT:
            raise ValueError("Agreement models must use the same labeled split")
        partner_models = restore_candidate(args.agreement_reference / "full" / CORRELATION_VARIANT,
                                           CORRELATION_VARIANT, CorrelationCNN, device)
        partner_threshold = partner_state["threshold"]
    policy = agreeing_blend if partner_models is not None else combine
    report = {"split_sha256": identity, "policy": policy.__name__, "selection_count": len(selection),
              "confirmation_count": len(confirmation), "screen": {}, "confirmation": {}}

    def record(stage):
        report["stage"] = stage
        write_json(args.output / "summary.json", report)

    def predictions(items, profile):
        native, codec, canonical, guard = compare_views(items, profile, table, deadline)
        rows = torch.arange(len(items))
        before = released.score(native, codec, guard, rows, deadline)
        after = recenter(scores(models, canonical, rows, device, deadline), state["threshold"])
        partner = (recenter(scores(partner_models, canonical, rows, device, deadline), partner_threshold)
                   if partner_models is not None else None)
        return before, after, partner

    def decision(before, after, partner, weight):
        if partner is not None:
            return agreeing_blend(before, after, partner, weight)
        return combine(before, after, weight)

    def verify(indices, profiles, weights, section):
        items, truth = [samples[i] for i in indices], labels[indices].numpy()
        pairs = {weight: [] for weight in weights}
        for profile in profiles:
            record(f"validate_{section}_{profile}")
            before, after, partner = predictions(items, profile)
            for weight in weights:
                refined = decision(before, after, partner, weight)
                pairs[weight].append((before, refined))
                measured = comparison(truth, before, refined)
                report[section].setdefault(profile, {})[str(weight)] = measured
                print(f"{section} {profile} w={weight:g}: {measured['reference']['accuracy']:.2%} -> "
                      f"{measured['deep_fusion']['accuracy']:.2%}", flush=True)
        return {weight: profile_gain(truth, changes) for weight, changes in pairs.items()}

    profiles = ("native", *CONFIRM)
    if args.verify_specialist:
        if state["stage"] != "stopped_at_screen" or state.get("selection_validation_count") != len(selection):
            raise ValueError("Expected the full resize specialist and its completed selection split")
        weight = 0.25
        if any(row[str(weight)]["delta"]["accuracy"] <= 0 for row in state["screen"].values()):
            raise ValueError("The fixed specialist weight must improve every selection profile")
        report["screen_gain"] = state["screen_gain"]
        report["screen"] = state["screen"]
        profiles = ("native", "jpeg76", "alternate_resize_jpeg", "webp")
    else:
        weights = (0.25, 0.5, 0.75) if partner_models is not None else (0.25, *WEIGHTS)
        trials = verify(selection, SCREEN, weights, "screen")
        report["screen_gain"] = trials
        eligible = [weight for weight, result in trials.items()
                    if result["mean_gain"] >= 0.025 and result["paired_95_percent"][0] > 0
                    and all(row[str(weight)]["delta"]["accuracy"] >= -0.015 for row in report["screen"].values())]
        if not eligible:
            record("screen_fell_short")
            return
        weight = max(eligible, key=lambda value: trials[value]["mean_gain"])
    if partner_models is not None:
        profiles = ("native", "jpeg76", "alternate_resize_jpeg", "resize_webp", "webp")
    report["selected_weight"] = weight
    confirmed = verify(confirmation, profiles, (weight,), "confirmation")[weight]
    report["confirmation_gain"] = confirmed
    insufficient_gain = (args.verify_specialist or partner_models is not None) and confirmed["mean_gain"] < 0.025
    if (confirmed["paired_95_percent"][0] <= 0 or insufficient_gain or
            any(row[str(weight)]["delta"]["accuracy"] < -0.01 for row in report["confirmation"].values())):
        record("confirmation_fell_short")
        return

    record("export")
    submission = args.output / "submission.csv"
    test = read_test(Path("ai-generated-image-detection-benchmark/sample_submission.csv"), Path("dataset/dataset/test"))
    if len(test) != 1000:
        raise ValueError("Expected a 1,000-image submission")
    before, after, partner = predictions(test, "native")
    refined = decision(before, after, partner, weight)
    write_csv(submission, ("image_id", "label"),
              ((item.image_id, int(score > 0.5)) for item, score in zip(test, refined, strict=True)))
    report["csv_checks"] = assess_export(submission)
    feasible = report["csv_checks"]["best_possible_correct"] >= TARGET_CORRECT
    record("candidate_ready" if feasible else "cannot_reach_target")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--correlation", type=Path, default=Path("lqy/runs/codec-canonical/correlation-full-01"))
    parser.add_argument("--verify-specialist", action="store_true",
                        help="Verify a fixed resize specialist blend on the confirmation split")
    parser.add_argument("--agreement-reference", type=Path, help="Use a second candidate when both models agree")
    parser.add_argument("--narrow-reference", type=Path, default=Path("lqy/runs/codec-final/upstream-trained"))
    parser.add_argument("--near-reference", type=Path, default=Path("lqy/runs/codec-final/near-upstream"))
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    if args.verify_specialist and args.agreement_reference is not None:
        parser.error("Choose either fixed specialist validation or agreement validation")
    with run_lock(args.output):
        run(args, device_for(args.device, 6), time.monotonic() + 2700)


if __name__ == "__main__":
    main()

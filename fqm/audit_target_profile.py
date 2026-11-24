"""Paired geometry/encoding audit and common held-out scorecard."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from audit_proxy_shift import ROOT, RUNS, TUNE, degrade
from ssd.data import ImageDataset, build_transform, load_norm_stats, read_splits
from ssd.engine import compute_metrics, load_checkpoint, predict, resolve_amp
from ssd.target_profile import FixedTargetMatch, validation_groups
from ssd.tta import TTATransform, evaluate_tta
from train import build_model

CASES = ("orig", "geometry", "encoding", "both", "legacy_strong")
SMOOTHED_CASE = "both_blur"


@torch.inference_mode()
def run(config, checkpoint, rows, all_rows, profiles, cases, device, output, seed, five_crop=False):
    data = config["data"]
    stats = load_norm_stats(ROOT / data["norm_stats"])
    tf = build_transform(data["input_size"], data["resize_size"], stats["mean"],
                         stats["std"], train=False, aug=data.get("aug", {}))
    if five_crop:
        tf = TTATransform(data["input_size"], data["resize_size"], stats["mean"], stats["std"],
                          mode=data.get("aug", {}).get("mode", "protocol"), flip=False)
    model = build_model(config["model"]).to(device)
    load_checkpoint(checkpoint, model, device)
    model.eval()
    amp, dtype = resolve_amp(config["train"]["amp"], config["train"]["amp_dtype"], device)
    y = np.array([row["label"] for row in rows])
    ids = [row["image_id"] for row in rows]
    scoring, probabilities = {}, {}
    for case in cases:
        smoothed = case == SMOOTHED_CASE
        assignment = FixedTargetMatch(all_rows, profiles, "both" if smoothed else case, seed,
                                      other_blur_radius=.35 if smoothed else 0.)
        if case == "orig":
            change = None
        elif case == "legacy_strong":
            change = lambda image, image_id: degrade(image, image_id, "strong")
        else:
            change = assignment
        dataset = ImageDataset(rows, tf, change)
        loader = DataLoader(dataset, batch_size=8 if five_crop else 32, num_workers=2,
                            pin_memory=device.type == "cuda")
        probs = (evaluate_tta(model, loader, device, amp, dtype)["probs"] if five_crop
                 else predict(model, loader, device, amp, dtype))
        cohorts = np.array([assignment.cohort(image_id) for image_id in ids])
        scores = {"all": compute_metrics(y, probs)}
        for cohort in ("width500", "other"):
            keep = cohorts == cohort
            scores[cohort] = compute_metrics(y[keep], probs[keep])
        scoring[case] = scores
        probabilities[case] = probs[:, 1]
        print(checkpoint.parent.name, case, {key: round(value["accuracy"], 4)
                                             for key, value in scores.items()}, flush=True)
    with output.with_suffix(".json").open("x") as file:
        json.dump(scoring, file, ensure_ascii=False, indent=2)
    with output.with_suffix(".npz").open("xb") as file:
        np.savez(file, ids=ids, labels=y, cohorts=cohorts, **probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--five-crop", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=(*CASES, SMOOTHED_CASE), default=CASES)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    profiles = json.loads(args.profiles.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in args.models:
        config = json.loads((TUNE / f"{name}.json").read_text())
        data = config["data"]
        _, val = read_splits(ROOT / data["splits"], ROOT / data["image_dir"])
        rows = validation_groups(val)[1] if args.heldout else val
        run(config, RUNS / name / "best.pt", rows, val, profiles, args.cases,
            device, args.output / name, seed=126, five_crop=args.five_crop)


if __name__ == "__main__":
    main()

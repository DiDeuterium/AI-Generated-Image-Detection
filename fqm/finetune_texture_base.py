"""Compare two full fine-tuning recipes from a shared self-supervised checkpoint."""

import argparse
from pathlib import Path
import time

import torch

from detection.training import load_checkpoint
from lqy.canonical_augmentation import canonical_views
from lqy.codec_patch import labeled_data
from lqy_resize_retrain import training_split
from native_texture_base import NativeTextureBase, SEED, acquisition, fit
from two_stage_texture_base import (
    FINETUNED_MODEL, PRETEXT_EPOCHS, PRETEXT_MODEL, classifier_from_backbone, write_summary,
)
from lqy_replay import read_json


EPOCHS = 24
RECIPES = {"uniform": 1.0, "lower_backbone_lr": .3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pretext", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 70 * 60
    split, details = labeled_data()
    train, calibration = training_split(split["train"])
    source = read_json(args.pretext / "summary.json")
    if source["split_sha256"] != details["split_sha256"] or source["stage"] != "training_complete":
        raise ValueError("Expected a completed pretext checkpoint from the same split")
    checkpoint = args.pretext / f"epoch-{PRETEXT_EPOCHS}.pt"
    state = load_checkpoint(checkpoint)
    if (state["variant"], state["epoch"], state["seed"]) != (PRETEXT_MODEL, PRETEXT_EPOCHS, SEED):
        raise ValueError("Unexpected pretext checkpoint variant, epoch, or seed")

    table = source["codec_table"]
    views = canonical_views(train, table, deadline, paired=True, alter_both=True, acquisition=acquisition)
    labels = torch.tensor([row.label for row in train])
    for recipe, backbone_lr_scale in RECIPES.items():
        output = args.output / recipe
        output.mkdir()
        model = classifier_from_backbone(NativeTextureBase, state["weights"], device)
        report = {"variant": FINETUNED_MODEL, "recipe": recipe, "seed": SEED, "epochs": EPOCHS,
                  "split_sha256": details["split_sha256"], "train_count": len(train),
                  "calibration_count": len(calibration), "codec_table": table,
                  "pretext_checkpoint": str(checkpoint), "head_initial_lr": .001,
                  "backbone_initial_lr": .001 * backbone_lr_scale}
        write_summary(output, report, "training")
        print(f"Fine-tuning recipe: {recipe}", flush=True)
        fit(model, views, labels, output, device, EPOCHS, deadline, variant=FINETUNED_MODEL,
            backbone_lr_scale=backbone_lr_scale)
        write_summary(output, report, "training_complete")


if __name__ == "__main__":
    main()

"""Train on unlabeled texture targets, then compare a frozen encoder with fine-tuning."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from types import SimpleNamespace

from PIL import Image, JpegImagePlugin
import torch

from detection.training import seed_all
from lqy.codec_patch import labeled_data
from lqy_resize_retrain import training_split
from native_texture_base import NativeTextureBase, SEED, fit
from self_supervised_texture_base import (
    SelfSupervisedTexture, residual_autocorrelation, texture_batch, texture_loss, training_views,
)


PRETEXT_MODEL = "two_stage_texture_pretext"
FROZEN_MODEL = "two_stage_texture_frozen"
FINETUNED_MODEL = "two_stage_texture_finetuned"
PRETEXT_EPOCHS = 12
CLASSIFIER_EPOCHS = 12


class FrozenBackboneTexture(NativeTextureBase):
    def __init__(self):
        super().__init__()
        self.requires_grad_(False)
        self.classifier.requires_grad_(True)

    def train(self, mode=True):
        super().train(False)
        self.classifier.train(mode)
        return self


def jpeg_table(sample):
    with Image.open(sample.path) as image:
        if image.format == "JPEG" and JpegImagePlugin.get_sampling(image) in (0, 1, 2):
            tables = image.quantization
            if {0, 1}.issubset(tables):
                return tuple(tables[0]), tuple(tables[1])
    return None


def unlabeled_codec_table(image_rows):
    with ThreadPoolExecutor(max_workers=4) as workers:
        counts = Counter(table for table in workers.map(jpeg_table, image_rows) if table is not None)
    if not counts:
        raise ValueError("No compatible JPEG quantization tables in the training images")
    return counts.most_common(1)[0][0]


def pretext_loss(model, first, second, _labels):
    pixels, targets, active = (torch.cat((a, b)) for a, b in zip(first, second, strict=True))
    prediction = model.texture_head(model.texture_embeddings(pixels))
    return texture_loss(prediction, targets, active)


def classifier_from_backbone(model_class, pretrained, device):
    seed_all(SEED, device)
    model = model_class()
    state = model.state_dict()
    for name in state:
        if not name.startswith("classifier."):
            state[name] = pretrained[name]
    model.load_state_dict(state)
    return model.to(device, memory_format=torch.channels_last)


def write_summary(output, report, stage):
    (output / "summary.json").write_text(json.dumps({**report, "stage": stage}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 75 * 60
    split, details = labeled_data()
    train, calibration = training_split(split["train"])
    image_rows = [SimpleNamespace(image_id=row.image_id, path=row.path) for row in train]
    table = unlabeled_codec_table(image_rows)
    report = {"seed": SEED, "split_sha256": details["split_sha256"], "train_count": len(image_rows),
              "calibration_count": len(calibration), "pretext_epochs": PRETEXT_EPOCHS,
              "gpu": torch.cuda.get_device_name(device), "codec_table": table}

    pretext = args.output / "pretext"
    pretext.mkdir()
    pretext_report = {**report, "variant": PRETEXT_MODEL, "epochs": PRETEXT_EPOCHS,
                      "target": "pre-acquisition luminance residual autocorrelation at lags one and two"}
    write_summary(pretext, pretext_report, "training")
    seed_all(SEED, device)
    encoder = SelfSupervisedTexture().to(device, memory_format=torch.channels_last)
    encoder.classifier.requires_grad_(False)
    encoder.attention.requires_grad_(False)
    packed = training_views(image_rows, table, deadline, target_fn=residual_autocorrelation)
    print("Self-supervised stage without class labels", flush=True)
    fit(encoder, packed, None, pretext, device, PRETEXT_EPOCHS, deadline, variant=PRETEXT_MODEL,
        get_batch=texture_batch, loss_fn=pretext_loss, sample_count=len(image_rows))
    write_summary(pretext, pretext_report, "training_complete")
    pretrained = {name: value.detach().cpu() for name, value in encoder.state_dict().items()}
    del encoder
    views = tuple(pixels for pixels, _, _ in packed)
    del packed
    labels = torch.tensor([row.label for row in train])

    for name, variant, model_class in (("frozen", FROZEN_MODEL, FrozenBackboneTexture),
                                      ("finetuned", FINETUNED_MODEL, NativeTextureBase)):
        output = args.output / name
        output.mkdir()
        model = classifier_from_backbone(model_class, pretrained, device)
        stage_report = {**report, "variant": variant, "epochs": CLASSIFIER_EPOCHS,
                        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                        "pretext_checkpoint": str(pretext / f"epoch-{PRETEXT_EPOCHS}.pt")}
        write_summary(output, stage_report, "training")
        print(f"Supervised stage: {name}", flush=True)
        fit(model, views, labels, output, device, CLASSIFIER_EPOCHS, deadline, variant=variant)
        write_summary(output, stage_report, "training_complete")


if __name__ == "__main__":
    main()

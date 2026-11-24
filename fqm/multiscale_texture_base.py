"""Train one shared CNN to classify native texture at two patch sizes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch import nn

from detection.data import jpeg_roundtrip
from detection.training import seed_all
from lqy.canonical_augmentation import CANONICAL, TRAIN_CROPS
from lqy.codec_patch import encode, labeled_data, tables_by_frequency
from lqy.night_train import check_time, image_patches
from lqy.spectrum_patch import QUALITY
from lqy_resize_retrain import training_split
from native_texture_base import NativeTextureBase, SEED, acquisition, consistency_loss, fit

MODEL = "multiscale_texture_base"
SCALES = (96, 160)


class MultiscaleTexture(NativeTextureBase):
    def __init__(self):
        super().__init__()
        channels = self.attention.in_features
        self.scale_fusion = nn.Linear(2 * channels, channels, bias=False)
        nn.init.zeros_(self.scale_fusion.weight)

    def forward(self, small, large):
        local = self.pool_patches(small)
        context = self.pool_patches(large)
        cross_scale = self.scale_fusion(torch.cat((local, context), dim=1))
        return self.classifier(.5 * (local + context) + .5 * cross_scale)


def native_scales(image, table):
    image = jpeg_roundtrip(image, quality=CANONICAL, subsampling=2)
    encoder = partial(encode, table=table)
    return tuple(np.concatenate((
        image_patches(image, QUALITY, crop_size=size),
        image_patches(image, QUALITY, encoder=encoder, crop_size=size),
    ), axis=1) for size in SCALES)


def render_training(sample, table):
    with Image.open(sample.path) as image:
        rgb = image.convert("RGB")
    assignment = sha256(f"canonical-augmentation:{sample.image_id}".encode()).digest()
    return tuple(native_scales(acquisition(rgb, sequence), table)
                 for sequence in (assignment[5:], assignment))


def training_views(samples, table, deadline):
    crops = list(TRAIN_CROPS)
    cache = tuple(tuple(np.empty((len(samples), len(crops), 6, size, size), dtype=np.uint8)
                        for size in SCALES) for _ in range(2))
    render = partial(render_training, table=table)
    with ThreadPoolExecutor(max_workers=4) as threads:
        for start in range(0, len(samples), 32):
            check_time(deadline)
            for index, views in enumerate(threads.map(render, samples[start:start + 32]), start=start):
                for stored, pair in zip(cache, views, strict=True):
                    for scale, patches in zip(stored, pair, strict=True):
                        scale[index] = patches[crops]
            if (start + 32) % 640 == 0 or start + 32 >= len(samples):
                print(f"Cached {min(start + 32, len(samples))}/{len(samples)} training images", flush=True)
    return tuple(tuple(torch.from_numpy(scale) for scale in view) for view in cache)


def batch_by_scale(view, ids, device):
    return tuple(scale[ids].to(device) for scale in view)


def paired_loss(model, first, second, labels):
    small, large = (torch.cat((a, b)) for a, b in zip(first, second, strict=True))
    a, b = model(small, large).float().chunk(2)
    return consistency_loss(a, b, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", default=24, type=int)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 85 * 60
    groups, details = labeled_data()
    train, calibration = training_split(groups["train"])
    tables, _ = tables_by_frequency(groups["train"])
    seed_all(SEED, device)
    model = MultiscaleTexture().to(device, memory_format=torch.channels_last)
    report = {"variant": MODEL, "gpu": torch.cuda.get_device_name(device), "seed": SEED,
              "epochs": args.epochs, "patch_sizes": SCALES, "split_sha256": details["split_sha256"],
              "train_count": len(train), "calibration_count": len(calibration),
              "parameters": sum(param.numel() for param in model.parameters()), "stage": "training"}
    summary = args.output / "summary.json"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    views = training_views(train, tables[0], deadline)
    targets = torch.tensor([row.label for row in train])
    fit(model, views, targets, args.output, device, args.epochs, deadline,
        variant=MODEL, get_batch=batch_by_scale, loss_fn=paired_loss)
    report["stage"] = "training_complete"
    summary.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

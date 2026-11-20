"""Train and evaluate the native-pixel ResNet-18 experiment."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18
from torchvision.transforms import functional as TF

from detection.data import EpochSampler, estimate_normalization, fingerprint, jpeg_roundtrip, make_loader
from detection.training import fit, load_checkpoint, seed_all, write_json

from . import tta
from .cnn import REPO, device_for, environment, load_samples, read_config, read_json, run_lock

MODEL = "resnet18-native-haar"


def pad_to_crop(image, size):
    width, height = image.size
    horizontal = max(0, size - width)
    vertical = max(0, size - height)
    if not (horizontal or vertical):
        return image
    pixels = np.pad(np.asarray(image), (
        (vertical // 2, vertical - vertical // 2),
        (horizontal // 2, horizontal - horizontal // 2),
        (0, 0),
    ), mode="reflect")
    return Image.fromarray(pixels)


class NativeCrops:
    def __init__(self, samples, config, *, train=False, jpeg=False, five=False):
        self.samples = samples
        self.config = config
        self.train = train
        self.jpeg = jpeg
        self.five = five

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rng = None
        if self.train:
            epoch_seed, index = index
            rng = random.Random((epoch_seed << 32) + index)
        sample = self.samples[index]
        with Image.open(sample.path) as source:
            image = source.convert("RGB")
        cfg = self.config
        if self.jpeg or (rng is not None and rng.random() < cfg["comparison_jpeg_probability"]):
            image = jpeg_roundtrip(image, cfg["jpeg_quality"], cfg["jpeg_subsampling"])
        size = cfg["crop_size"]
        image = pad_to_crop(image, size)
        if self.five:
            pixels = torch.stack([TF.pil_to_tensor(crop) for crop in TF.five_crop(image, size)])
        elif self.train:
            width, height = image.size
            top = rng.randrange(height - size + 1)
            left = rng.randrange(width - size + 1)
            crop = TF.crop(image, top, left, size, size)
            if rng.random() < cfg["flip_probability"]:
                crop = TF.hflip(crop)
            pixels = TF.pil_to_tensor(crop)
        else:
            pixels = TF.pil_to_tensor(TF.center_crop(image, size))
        return pixels, sample.label, sample.image_id


def five_crops(samples, config, _norm, *, jpeg):
    return NativeCrops(samples, config, jpeg=jpeg, five=True)


class NativeResNet18(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.net = resnet18(weights=None, num_classes=2)
        self.net.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.register_buffer("mean", torch.tensor(norm["mean"]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(norm["std"]).view(1, 3, 1, 1))

    def forward(self, pixels):
        # The four coefficients retain the mean and directional details of each native 2x2 tile.
        rgb = pixels.float() * (1 / (4 * 255))
        a, b = rgb[:, :, ::2, ::2], rgb[:, :, ::2, 1::2]
        c, d = rgb[:, :, 1::2, ::2], rgb[:, :, 1::2, 1::2]
        features = torch.cat(((a + b + c + d - self.mean) / self.std,
                              16 * (a - b + c - d),
                              16 * (a + b - c - d),
                              16 * (a - b - c + d)), dim=1)
        return self.net(features)


def build_model(metadata, device):
    if metadata["model"] != MODEL:
        raise ValueError(f"Expected a {MODEL} checkpoint")
    memory_format = torch.channels_last if device.type == "xpu" else torch.contiguous_format
    return NativeResNet18(metadata["norm"]).to(device, memory_format=memory_format)


def native_loader(dataset, config, args, device, *, sampler=None):
    return make_loader(dataset, config["microbatch"], args.workers, sampler=sampler,
                       persistent=bool(args.workers), pin_memory=device.type == "cuda")


def normalization(path, samples, identity, config, args, device):
    identity = {**identity, "geometry": "native-crop-reflect-v1"}
    if path.exists():
        norm = read_json(path)
        if any(norm.get(key) != value for key, value in identity.items()):
            raise ValueError("Cached normalization uses a different split or crop setup")
        return norm
    dataset = NativeCrops(samples, config)
    raw = estimate_normalization(native_loader(dataset, config, args, device))
    norm = {**identity, **raw}
    for key in ("mean", "std"):
        norm[key] = [value / 255 for value in raw[key]]
    write_json(path, norm)
    return norm


def train(args, device):
    config = read_config(args)
    if config["crop_size"] % 2 or not 0 <= config.get("ema_decay", 0) < 1:
        raise ValueError("Use an even crop size and EMA decay in [0, 1)")
    groups, identity = load_samples(args, config)
    with run_lock(args.output):
        norm = normalization(args.output / "normalization.json", groups["train"], identity,
                             config, args, device)
        train_data = NativeCrops(groups["train"], config, train=True)
        sampler = EpochSampler(train_data)
        train_batches = native_loader(train_data, config, args, device, sampler=sampler)
        val = native_loader(NativeCrops(groups["val"], config), config, args, device)
        jpeg = native_loader(NativeCrops(groups["val"], config, jpeg=True), config, args, device)
        versions = environment(device, args)
        versions["source_sha256"]["lqy/native.py"] = fingerprint(Path(__file__).resolve())
        for seed in args.seeds:
            seed_all(seed, device)
            model = build_model({"model": MODEL, "norm": norm}, device)
            metadata = {
                "model": MODEL, "params": sum(parameter.numel() for parameter in model.parameters()),
                "seed": seed, "config": config, "norm": norm, "environment": versions,
                "train_count": len(train_data), "val_count": len(groups["val"]),
                "jpeg_train_probability": config["comparison_jpeg_probability"],
            }
            run = args.output / f"resnet18-native-s{seed}"
            with run_lock(run):
                if (run / "summary.json").exists() and (run / "best.pt").exists():
                    summary = read_json(run / "summary.json")
                    if all(summary.get(key) == value for key, value in metadata.items()):
                        continue
                write_json(run / "config.json", config)
                fit(model, lambda epoch_seed: epoch_loader(epoch_seed, sampler, train_batches),
                    val, jpeg, device, run, metadata, resume=(run / "last.pt").exists(),
                    epoch_limit=args.epoch_limit)


def epoch_loader(seed, sampler, provider):
    sampler.seed = seed
    return provider


def preview(args, device):
    state = load_checkpoint(args.run / "last.pt")
    metadata = state["metadata"]
    weights = {"best": state["best"]["model_state"]}
    if args.compare_ema:
        if "ema" not in state:
            raise ValueError("The run has no EMA weights to compare")
        weights["last-raw"] = state["model_state"]
        weights["last-ema"] = {name.removeprefix("module."): tensor
                               for name, tensor in state["ema"].items()
                               if name.startswith("module.")}
    models = {}
    for name, weight in weights.items():
        model = build_model(metadata, device)
        model.load_state_dict(weight)
        models[name] = model.eval()
    print(f"Previewing best epoch {state['best']['epoch']}", flush=True)
    tta.evaluate(args, models, metadata["config"], metadata["norm"], device, five_crops)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--batch-size", type=int, default=12)
    cli.add_argument("--forward-batch", type=int, default=32)
    commands = cli.add_subparsers(dest="command", required=True)
    train_parser, preview_parser, val_parser, export_parser = [
        commands.add_parser(name) for name in ("train", "preview", "evaluate", "export")
    ]
    for command in (train_parser, preview_parser, val_parser, export_parser):
        command.add_argument("--output", type=Path, required=True)
    for command in (train_parser, preview_parser, val_parser):
        command.add_argument("--manifest", type=Path, default=Path("splits/train_val.csv"))
        command.add_argument("--image-root", type=Path, default=Path("dataset"))
    train_parser.add_argument("--config", type=Path, default=REPO / "lqy" / "native320.json")
    train_parser.add_argument("--microbatch", type=int)
    train_parser.add_argument("--seeds", type=int, nargs="+", default=[42, 17, 43, 97])
    train_parser.add_argument("--epoch-limit", type=int)
    preview_parser.add_argument("--run", type=Path, required=True)
    preview_parser.add_argument("--compare-ema", action="store_true")
    for command in (val_parser, export_parser):
        command.add_argument("--runs", type=Path, nargs="+", required=True)
    export_parser.add_argument("--test-dir", type=Path, default=Path("dataset/dataset/test"))
    export_parser.add_argument("--template", type=Path,
                               default=Path("ai-generated-image-detection-benchmark/sample_submission.csv"))
    args = cli.parse_args()
    try:
        if args.workers < 0 or min(args.batch_size, args.forward_batch) < 1:
            raise ValueError("Worker count must be non-negative and batch sizes must be positive")
        device = device_for(args.device, args.threads)
        if args.command == "train":
            if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
                raise ValueError("Seed values must be distinct and non-negative")
            if args.epoch_limit is not None and args.epoch_limit < 1:
                raise ValueError("Epoch limit must be positive")
            train(args, device)
        elif args.command == "preview":
            preview(args, device)
        else:
            if len({run.name for run in args.runs}) != len(args.runs):
                raise ValueError("Run directory names must be distinct")
            models, config, norm = tta.load_models(args.runs, device, model_for=build_model)
            action = tta.evaluate if args.command == "evaluate" else tta.export
            action(args, models, config, norm, device, five_crops)
    except (ValueError, FileNotFoundError) as exc:
        cli.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

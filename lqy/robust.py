"""Train and check an augmented native-pixel member A ResNet-18."""

import argparse
import hashlib
import random
from pathlib import Path

import torch
from PIL import Image, ImageFilter, ImageOps
from torch import nn
from torchvision.transforms import functional as TF

from detection.data import EpochSampler, fingerprint, jpeg_roundtrip, make_loader
from detection.metrics import binary_metrics
from detection.training import fit, seed_all, write_csv, write_json

from . import native, tta
from .cnn import REPO, device_for, environment, load_samples, read_config, read_json, run_lock


AUGMENTATION = {
    "resize_probability": 0.8,
    "short_sides": [288, 352, 416, 480, 544, 608],
    "aspect_ratios": [1.0, 6 / 5, 4 / 3, 3 / 2, 5 / 3],
    "resampling": ["BILINEAR", "BICUBIC", "LANCZOS"],
    "jpeg_probability": 0.85,
    "jpeg_quality_range": [60, 95],
    "blur_probability": 0.12,
    "blur_radius_range": [0.15, 0.8],
}
STRESS = {
    "landscape_jpeg_q70": ((448, 336), Image.Resampling.BICUBIC, 70),
    "portrait_jpeg_q90": ((384, 512), Image.Resampling.LANCZOS, 90),
}
TEXTURE_MODEL = "resnet18-native-texture"


class TextureResNetHead(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.early = nn.Conv2d(64, 16, kernel_size=1, bias=False)
        self.middle = nn.Conv2d(128, 24, kernel_size=1, bias=False)
        self.texture = nn.Sequential(nn.LayerNorm(16 ** 2 + 24 ** 2), nn.Linear(16 ** 2 + 24 ** 2, 128),
                                     nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(512 + 128, 2))

    @staticmethod
    def texture_correlation(features):
        channels = features.flatten(2).float()
        centered = channels - channels.mean(dim=2, keepdim=True)
        scale = (centered.square().mean(dim=2, keepdim=True) + 1e-4).rsqrt()
        normalized = centered * scale
        return (normalized @ normalized.transpose(1, 2) / normalized.shape[2]).flatten(1)

    def forward(self, features):
        backbone = self.backbone
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(features))))
        early = backbone.layer1(x)
        middle = backbone.layer2(early)
        late = backbone.layer4(backbone.layer3(middle))
        texture = self.texture(torch.cat((self.texture_correlation(self.early(early)),
                                          self.texture_correlation(self.middle(middle))), dim=1))
        return self.classifier(torch.cat((backbone.avgpool(late).flatten(1), texture), dim=1))


def build_model(metadata, device):
    if metadata["model"] == native.MODEL:
        return native.build_model(metadata, device)
    if metadata["model"] != TEXTURE_MODEL:
        raise ValueError(f"Unsupported model {metadata['model']}")
    model = native.NativeResNet18(metadata["norm"])
    model.net = TextureResNetHead(model.net)
    memory_format = torch.channels_last if device.type == "xpu" else torch.contiguous_format
    return model.to(device, memory_format=memory_format)


class AugmentedCrops(native.NativeCrops):
    def __getitem__(self, key):
        epoch_seed, index = key
        sample = self.samples[index]
        rng = random.Random((epoch_seed << 32) + index)
        with Image.open(sample.path) as source:
            image = source.convert("RGB")

        aug = self.config["augmentation"]
        if rng.random() < aug["resize_probability"]:
            short = rng.choice(aug["short_sides"])
            long = round(short * rng.choice(aug["aspect_ratios"]))
            size = rng.choice(((short, long), (long, short)))
            method = Image.Resampling[rng.choice(aug["resampling"])]
            center = (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))
            image = ImageOps.fit(image, size, method=method, centering=center)
        if rng.random() < aug["blur_probability"]:
            image = image.filter(ImageFilter.GaussianBlur(rng.uniform(*aug["blur_radius_range"])))
        if rng.random() < aug["jpeg_probability"]:
            quality = rng.randint(*aug["jpeg_quality_range"])
            image = jpeg_roundtrip(image, quality, self.config["jpeg_subsampling"])

        size = self.config["crop_size"]
        image = native.pad_to_crop(image, size)
        width, height = image.size
        top = rng.randrange(height - size + 1)
        left = rng.randrange(width - size + 1)
        crop = TF.crop(image, top, left, size, size)
        if rng.random() < self.config["flip_probability"]:
            crop = TF.hflip(crop)
        return TF.pil_to_tensor(crop), sample.label, sample.image_id


class StressCrops(native.NativeCrops):
    def __init__(self, samples, config, profiles, *, five=False):
        super().__init__(samples, config, five=five)
        self.profiles = profiles

    def __getitem__(self, index):
        sample = self.samples[index]
        size, method, quality = self.profiles[index % len(self.profiles)]
        with Image.open(sample.path) as source:
            image = ImageOps.fit(source.convert("RGB"), size, method=method)
        image = jpeg_roundtrip(image, quality, self.config["jpeg_subsampling"])
        crop_size = self.config["crop_size"]
        if self.five:
            pixels = torch.stack([TF.pil_to_tensor(crop) for crop in TF.five_crop(image, crop_size)])
        else:
            pixels = TF.pil_to_tensor(TF.center_crop(image, crop_size))
        return pixels, sample.label, sample.image_id


def validation_halves(samples):
    def order(sample):
        return hashlib.sha256(f"course-validation:{sample.image_id}".encode()).digest()

    selection, holdout = [], []
    for label in (0, 1):
        group = sorted((sample for sample in samples if sample.label == label), key=order)
        midpoint = len(group) // 2
        selection.extend(group[:midpoint])
        holdout.extend(group[midpoint:])
    return selection, holdout


def labeled_splits(args, config):
    groups, identity = load_samples(args, config)
    root = args.image_root / "dataset/train"
    if any(not sample.path.is_relative_to(root) for samples in groups.values() for sample in samples):
        raise ValueError("Training and validation must both use the labeled training directory")
    return groups, identity


def train(args, device):
    config = read_config(args)
    if args.epochs < 2 or args.seed < 0 or config["crop_size"] != 320:
        raise ValueError("Use a 320-pixel crop, at least two epochs and a non-negative seed")
    config.update(epochs=args.epochs, warmup_updates=100, min_lr=3e-6,
                  max_lr=3e-4, augmentation=AUGMENTATION,
                  stress_profiles=list(STRESS))
    groups, identity = labeled_splits(args, config)
    selection, _ = validation_halves(groups["val"])
    with run_lock(args.output):
        norm = native.normalization(args.output / "normalization.json", groups["train"],
                                    identity, config, args, device)
        seed_all(args.seed, device)
        model = build_model({"model": TEXTURE_MODEL, "norm": norm}, device)

        dataset = AugmentedCrops(groups["train"], config)
        sampler = EpochSampler(dataset)
        batches = native.native_loader(dataset, config, args, device, sampler=sampler)
        original = native.native_loader(native.NativeCrops(groups["val"], config), config, args, device)
        jpeg = native.native_loader(native.NativeCrops(groups["val"], config, jpeg=True), config, args, device)
        stress = native.native_loader(StressCrops(selection, config, tuple(STRESS.values())),
                                     config, args, device)
        versions = environment(device, args)
        for name in ("lqy/native.py", "lqy/robust.py"):
            versions["source_sha256"][name] = fingerprint(REPO / name)
        metadata = {"model": TEXTURE_MODEL, "params": sum(p.numel() for p in model.parameters()),
                    "seed": args.seed, "config": config, "norm": norm, "environment": versions,
                    "train_count": len(dataset), "val_count": len(groups["val"]),
                    "stress_selection_count": len(selection), "initialization": "random"}
        summary_path = args.output / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            if (args.output / "best.pt").is_file() and all(summary.get(key) == value
                                                           for key, value in metadata.items()):
                print(f"Already completed {args.output}", flush=True)
                return
        else:
            write_json(args.output / "config.json", config)
        fit(model, lambda seed: native.epoch_loader(seed, sampler, batches), original, jpeg,
            device, args.output, metadata, resume=(args.output / "last.pt").exists(),
            selection_loader=stress)


def audit(args, device):
    groups = {}
    for name, paths in (("reference", args.reference), ("candidate", args.candidate)):
        if paths:
            groups[name] = tta.load_models(paths, device, model_for=build_model)
    _, config, norm = groups["candidate"]
    splits, identity = labeled_splits(args, config)
    if any(norm.get(key) != value for key, value in identity.items()):
        raise ValueError("Audit data differ from the candidate's training split")
    _, holdout = validation_halves(splits["val"])
    ids = [sample.image_id for sample in holdout]
    labels = [sample.label for sample in holdout]
    datasets = {"original": native.five_crops(holdout, config, norm, jpeg=False)}
    for name, profile in STRESS.items():
        datasets[name] = StressCrops(holdout, config, (profile,), five=True)

    metrics, columns = {}, {}
    for name, dataset in datasets.items():
        loader = make_loader(dataset, args.batch_size, args.workers, pin_memory=device.type == "cuda")
        metrics[name] = {}
        for model_name, (models, model_config, model_norm) in groups.items():
            if model_config["crop_size"] != config["crop_size"] or model_norm != norm:
                raise ValueError("Compare models with the same crop size and normalization")
            result_ids, result_labels, scores = tta.infer(models, loader, device, args.forward_batch)
            if result_ids != ids or result_labels != labels:
                raise ValueError("Audit rows are misaligned")
            probabilities = tta.ensemble_scores(scores, models)
            columns[f"{name}:{model_name}"] = probabilities
            result = binary_metrics(labels, probabilities > 0.5, probabilities)
            metrics[name][model_name] = result
            print(f"{name} {model_name}: accuracy={result['accuracy']:.4f} "
                  f"AUROC={result['auroc']:.4f}", flush=True)
    write_json(args.output / "metrics.json", metrics)
    write_csv(args.output / "val_scores.csv", ("image_id", "label", *columns),
              zip(ids, labels, *columns.values(), strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu", choices=("xpu", "cuda", "cpu", "auto"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", type=Path, default=REPO / "lqy/native320.json")
    training.add_argument("--epochs", type=int, default=18)
    training.add_argument("--microbatch", type=int)
    training.add_argument("--seed", type=int, default=42)
    validation = commands.add_parser("audit")
    validation.add_argument("--candidate", type=Path, nargs="+", required=True)
    validation.add_argument("--reference", type=Path, nargs="+")
    export = commands.add_parser("export")
    export.add_argument("--runs", type=Path, nargs="+", required=True)
    export.add_argument("--template", type=Path,
                        default=Path("ai-generated-image-detection-benchmark/sample_submission.csv"))
    export.add_argument("--test-dir", type=Path, default=Path("dataset/dataset/test"))
    for command in (validation, export):
        command.add_argument("--batch-size", type=int, default=12)
        command.add_argument("--forward-batch", type=int, default=32)
    for command in (training, validation):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--manifest", type=Path, default=Path("splits/train_val.csv"))
        command.add_argument("--image-root", type=Path, default=Path("dataset"))
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = device_for(args.device, args.threads)
    if args.command == "export":
        models, config, norm = tta.load_models(args.runs, device, model_for=build_model)
        tta.export(args, models, config, norm, device, native.five_crops)
    else:
        (train if args.command == "train" else audit)(args, device)


if __name__ == "__main__":
    main()

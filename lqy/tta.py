"""Evaluate five crops and their horizontal flips using trained CNN checkpoints."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

from detection.data import ImageDataset, make_loader, read_test
from detection.metrics import binary_metrics
from detection.training import load_checkpoint, move_images, write_csv, write_json

from .cnn import build_model, device_for, load_samples, read_json


class StackCrops:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, crops):
        return torch.stack([self.transform(crop) for crop in crops])


def five_crop_dataset(samples, config, norm, *, jpeg=False):
    dataset = ImageDataset(
        samples, config["short_side"], config["crop_size"], norm=norm,
        jpeg_probability=float(jpeg), quality=config["jpeg_quality"],
        subsampling=config["jpeg_subsampling"],
    )
    dataset.center_crop = transforms.FiveCrop(config["crop_size"])
    dataset.transform = StackCrops(dataset.transform)
    return dataset


def load_models(runs, device, model_for=None):
    models = {}
    summaries = [read_json(run / "summary.json") for run in runs]
    config, norm = summaries[0]["config"], summaries[0]["norm"]
    for run, summary in zip(runs, summaries, strict=True):
        if summary["config"] != config or summary["norm"] != norm:
            raise ValueError("TTA members must share the training configuration and normalization")
        checkpoint = load_checkpoint(run / "best.pt")
        metadata = checkpoint["metadata"]
        for key in ("model", "seed", "config", "norm"):
            if metadata[key] != summary[key]:
                raise ValueError(f"{run}: checkpoint {key} differs from the run summary")
        if checkpoint["epoch"] != summary["best_epoch"]:
            raise ValueError(f"{run}: checkpoint epoch differs from the run summary")
        model = (build_model(metadata["model"], device) if model_for is None
                 else model_for(metadata, device))
        model.load_state_dict(checkpoint["model_state"])
        models[run.name] = model.eval()
    return models, config, norm


@torch.inference_mode()
def infer(models, loader, device, forward_batch):
    ids, labels = [], []
    scores = {name: [] for name in models}
    for images, targets, batch_ids in loader:
        batch, views, channels, height, width = images.shape
        images = move_images(images.reshape(-1, channels, height, width), device)
        for name, model in models.items():
            predictions = []
            for chunk in images.split(forward_batch):
                direct = model(chunk).float().softmax(dim=1)[:, 1]
                mirrored = model(chunk.flip(dims=(-1,))).float().softmax(dim=1)[:, 1]
                predictions.append((direct + mirrored) * 0.5)
            probabilities = torch.cat(predictions).reshape(batch, views).mean(dim=1)
            if not torch.isfinite(probabilities).all():
                raise ValueError(f"Non-finite scores from {name} at {batch_ids[0]}")
            scores[name].extend(probabilities.cpu().tolist())
        ids.extend(batch_ids)
        labels.extend(targets.tolist())
    return ids, labels, {name: np.asarray(values, dtype=np.float64) for name, values in scores.items()}


def ensemble_scores(scores, members):
    return np.mean([scores[name] for name in members], axis=0)


def evaluate(args, models, config, norm, device, dataset_factory=five_crop_dataset):
    groups, identity = load_samples(args, config)
    if any(norm.get(key) != value for key, value in identity.items()):
        raise ValueError("Validation split or image geometry differs from the checkpoints")
    members = tuple(models)
    expected_ids = [sample.image_id for sample in groups["val"]]
    expected_labels = [sample.label for sample in groups["val"]]
    metrics, predictions = {}, {}
    for variant, jpeg in (("original", False), ("jpeg", True)):
        loader = loader_for(args, groups["val"], config, norm, device, dataset_factory, jpeg=jpeg)
        ids, labels, scores = infer(models, loader, device, args.forward_batch)
        if ids != expected_ids or labels != expected_labels:
            raise ValueError(f"{variant} validation IDs or labels differ from the fixed split")
        metrics[variant] = {
            name: binary_metrics(labels, scores[name] > 0.5, scores[name]) for name in members
        }
        predictions[variant] = scores
        print(f"Evaluated {len(ids)} {variant} images across {len(members)} checkpoints", flush=True)

    columns = ("image_id", "label", *(f"{variant}:{name}"
              for variant in ("original", "jpeg") for name in members))
    rows = zip(expected_ids, expected_labels, *(predictions[variant][name]
               for variant in ("original", "jpeg") for name in members), strict=True)
    write_csv(args.output / "val_scores.csv", columns, rows)
    write_json(args.output / "metrics.json", {"views": 10, "members": list(members), **metrics})


def loader_for(args, samples, config, norm, device, dataset_factory, *, jpeg=False):
    return make_loader(dataset_factory(samples, config, norm, jpeg=jpeg),
                       args.batch_size, args.workers, pin_memory=device.type == "cuda")


def export(args, models, config, norm, device, dataset_factory=five_crop_dataset):
    samples = read_test(args.template, args.test_dir)
    if len(samples) != config["test_count"]:
        raise ValueError(f"Expected {config['test_count']} test images, found {len(samples)}")
    ids, _, scores = infer(models, loader_for(args, samples, config, norm, device, dataset_factory),
                           device, args.forward_batch)
    if ids != [sample.image_id for sample in samples]:
        raise ValueError("Inference IDs must follow the submission template")
    classes = (ensemble_scores(scores, models) > 0.5).astype(int)
    write_csv(args.output, ("image_id", "label"), zip(ids, classes, strict=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    cli.add_argument("--workers", type=int, default=4)
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--batch-size", type=int, default=12, help="Number of images per loader batch.")
    cli.add_argument("--forward-batch", type=int, default=32, help="Crop limit per model forward pass.")
    commands = cli.add_subparsers(dest="command", required=True)
    validation = commands.add_parser("evaluate")
    submission = commands.add_parser("export")
    for command in (validation, submission):
        command.add_argument("--runs", type=Path, nargs="+", required=True)
        command.add_argument("--output", type=Path, required=True)
    validation.add_argument("--manifest", type=Path, default=Path("splits/train_val.csv"))
    validation.add_argument("--image-root", type=Path, default=Path("dataset"))
    submission.add_argument("--template", type=Path,
                            default=Path("ai-generated-image-detection-benchmark/sample_submission.csv"))
    submission.add_argument("--test-dir", type=Path, default=Path("dataset/dataset/test"))
    args = cli.parse_args()
    try:
        if args.workers < 0 or min(args.batch_size, args.forward_batch) < 1:
            raise ValueError("Worker count must be non-negative and batch sizes must be positive")
        if len({run.name for run in args.runs}) != len(args.runs):
            raise ValueError("Run directory names must be distinct")
        device = device_for(args.device, args.threads)
        models, config, norm = load_models(args.runs, device)
        command = evaluate if args.command == "evaluate" else export
        command(args, models, config, norm, device)
    except (ValueError, FileNotFoundError) as exc:
        cli.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

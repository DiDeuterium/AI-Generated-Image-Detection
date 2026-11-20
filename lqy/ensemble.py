"""Evaluate an equal-weight ensemble of completed CNN runs."""

import argparse
from pathlib import Path

import numpy as np

from detection.data import make_loader, read_csv, read_test
from detection.metrics import binary_metrics
from detection.training import load_checkpoint, predict, write_csv, write_json

from .cnn import build_model, device_for, image_dataset, read_json


def validate(runs):
    columns = ("image_id", "label", "original_score", "jpeg_score")
    predictions = [read_csv(run / "val_predictions.csv", columns) for run in runs]
    first = predictions[0]
    keys = [(row["image_id"], row["label"]) for row in first]
    if any([(row["image_id"], row["label"]) for row in rows] != keys for rows in predictions[1:]):
        raise ValueError("Validation image IDs, labels or order differ across runs")
    labels = [int(row["label"]) for row in first]
    results = {}
    averaged = {}
    for version in ("original", "jpeg"):
        raw = np.asarray([[float(row[f"{version}_score"]) for row in rows] for rows in predictions])
        scores = raw.mean(axis=0)
        classes = (scores > 0.5).astype(int)
        results[version] = binary_metrics(labels, classes, scores)
        averaged[version] = (classes, scores)
    return results, zip(
        [row["image_id"] for row in first], labels, *averaged["original"], *averaged["jpeg"],
        strict=True,
    )


def export(args, summaries, runs):
    device = device_for(args.device, args.threads)
    config, norm = summaries[0]["config"], summaries[0]["norm"]
    samples = read_test(args.template, args.test_dir)
    if len(samples) != config["test_count"]:
        raise ValueError(f"Unexpected test image count: {len(samples)}")
    loader = make_loader(image_dataset(samples, config, norm=norm), args.batch_size, args.workers,
                         pin_memory=device.type == "cuda", persistent=True)
    ids = [sample.image_id for sample in samples]
    scores = []
    for run, summary in zip(runs, summaries, strict=True):
        checkpoint = load_checkpoint(run / "best.pt")
        if checkpoint["metadata"]["norm"] != norm or checkpoint["epoch"] != summary["best_epoch"]:
            raise ValueError(f"Checkpoint disagrees with {run / 'summary.json'}")
        model = build_model(summary["model"], device)
        model.load_state_dict(checkpoint["model_state"])
        prediction = predict(model, loader, device)
        if prediction["ids"] != ids:
            raise ValueError("Inference image IDs must follow template order")
        scores.append(prediction["scores"])
        del model
    classes = (np.mean(scores, axis=0) > 0.5).astype(int)
    write_csv(args.output / "submission.csv", ("image_id", "label"), zip(ids, classes, strict=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, nargs="+", required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--export", action="store_true", help="Predict the competition test split.")
    cli.add_argument("--test-dir", type=Path, default=Path("dataset/dataset/test"))
    cli.add_argument("--template", type=Path,
                     default=Path("ai-generated-image-detection-benchmark/sample_submission.csv"))
    cli.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    cli.add_argument("--threads", type=int, default=4)
    cli.add_argument("--workers", type=int, default=2)
    cli.add_argument("--batch-size", type=int, default=32)
    args = cli.parse_args()
    runs = [run.resolve() for run in args.runs]
    if len(runs) < 2 or len(set(runs)) != len(runs):
        cli.error("Provide at least two distinct completed run directories")
    try:
        summaries = [read_json(run / "summary.json") for run in runs]
        if any(item["norm"] != summaries[0]["norm"] for item in summaries[1:]):
            raise ValueError("Ensemble runs must use the same split and normalization")
        metrics, rows = validate(runs)
        write_csv(args.output / "val_predictions.csv",
                  ("image_id", "label", "original_class", "original_score", "jpeg_class", "jpeg_score"),
                  rows)
        write_json(args.output / "validation.json", {
            "members": [str(run) for run in runs],
            "method": "arithmetic_mean_of_ai_class_probabilities",
            **metrics,
        })
        if args.export:
            if args.workers < 0 or args.batch_size < 1:
                raise ValueError("--workers must be non-negative and --batch-size must be positive")
            export(args, summaries, runs)
    except (ValueError, FileNotFoundError) as exc:
        cli.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

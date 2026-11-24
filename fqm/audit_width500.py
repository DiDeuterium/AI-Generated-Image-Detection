"""Evaluate a width-500 proxy using the observed quantization tables for both classes."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, JpegImagePlugin

from audit_proxy_shift import ROOT, RUNS, TUNE, evaluate
from ssd.data import build_transform, load_norm_stats, read_splits
from ssd.engine import compute_metrics, load_checkpoint
from train import build_model


class Width500Transform:
    def __init__(self, transform, quantization):
        self.transform = transform
        self.quantization = quantization

    def __call__(self, image):
        height = max(8, round(image.height * 500 / image.width))
        resized = image.resize((500, height), Image.Resampling.BICUBIC)
        with io.BytesIO() as buffer:
            resized.save(buffer, format="JPEG", qtables=self.quantization, subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                return self.transform(encoded.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with Image.open(args.reference) as image:
        assert image.width == 500 and JpegImagePlugin.get_sampling(image) == 2
        tables = image.quantization
    signature = hashlib.sha256(repr(tables).encode()).hexdigest()[:16]
    assert signature == "93f1417788ab3c9b"
    results = {}
    for name in args.models:
        config = json.loads((TUNE / f"{name}.json").read_text())
        data = config["data"]
        stats = load_norm_stats(ROOT / data["norm_stats"])
        _, rows = read_splits(ROOT / data["splits"], ROOT / data["image_dir"])
        transform = build_transform(data["input_size"], data["resize_size"], stats["mean"],
                                    stats["std"], train=False, aug=data.get("aug", {}))
        model = build_model(config["model"]).to(device).eval()
        load_checkpoint(RUNS / name / "best.pt", model, device)
        inference = evaluate(model, rows, Width500Transform(transform, tables),
                             "orig", device, 32, 2, False)
        margin = inference["margin"]
        probability = 1 / (1 + np.exp(-margin))
        results[name] = compute_metrics(np.array([row["label"] for row in rows]),
                                        np.stack((1 - probability, probability), axis=1))
        print(name, results[name], flush=True)
    with args.output.open("x") as output:
        json.dump({"quantization": tables, "results": results}, output, indent=2)


if __name__ == "__main__":
    main()

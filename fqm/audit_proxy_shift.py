"""Evaluate frozen local checkpoints on paired, deterministic image degradations."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ssd.data import build_transform, load_norm_stats, read_splits, read_test_ids
from ssd.domain_shift import legacy_degrade as degrade
from ssd.engine import load_checkpoint
from train import build_model

ROOT = Path(__file__).resolve().parents[1]
TUNE = ROOT / "fqm/tune/configs"
RUNS = ROOT / "fqm/runs/tune"
CASES = ("orig", "q75", "scale75", "scale75_q75", "moderate", "strong")


class ProxyImages(Dataset):
    def __init__(self, rows, transform, case):
        self.rows = rows
        self.transform = transform
        self.case = case

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        return self.transform(degrade(image, row["image_id"], self.case))


@torch.inference_mode()
def evaluate(model, rows, transform, case, device, batch_size, workers, features):
    loader = DataLoader(ProxyImages(rows, transform, case), batch_size=batch_size,
                        num_workers=workers, pin_memory=device.type == "cuda")
    scores, embeddings, pooled = [], [], []
    backbone = getattr(model, "backbone", model)
    hook = backbone.head_drop.register_forward_hook(
        lambda module, inputs, output: pooled.append(output)
    ) if features else None
    try:
        for batch in loader:
            pooled.clear()
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(batch)
            scores.append((logits[:, 1] - logits[:, 0]).float().cpu().numpy())
            if features:
                if len(pooled) != 4:
                    raise RuntimeError("Expected four pooled feature branches")
                embeddings.append(torch.cat(pooled, dim=1).float().cpu().numpy())
    finally:
        if hook is not None:
            hook.remove()
    result = {"margin": np.concatenate(scores), "ids": [row["image_id"] for row in rows]}
    if features:
        result["embedding"] = np.concatenate(embeddings)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", choices=(0, 2), type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-model", default="w6_cap60_jpegaug")
    parser.add_argument("models", nargs="+")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)

    for name in args.models:
        config = json.loads((TUNE / f"{name}.json").read_text())
        data = config["data"]
        stats = load_norm_stats(ROOT / data["norm_stats"])
        train, val = read_splits(ROOT / data["splits"], ROOT / data["image_dir"])
        test = read_test_ids(
            ROOT / "ai-generated-image-detection-benchmark/sample_submission.csv",
            ROOT / data["image_dir"] / "dataset/test",
        )
        transform = build_transform(data["input_size"], data["resize_size"], stats["mean"],
                                    stats["std"], train=False, aug=data.get("aug", {}))
        model = build_model(config["model"]).to(device).eval()
        load_checkpoint(RUNS / name / "best.pt", model, device)
        extract = name == args.embedding_model
        tasks = [(f"val_{case}", val, case, extract and case in ("orig", "q75", "moderate", "strong"))
                 for case in CASES]
        tasks.append(("test_orig", test, "orig", extract))
        if extract:
            tasks.extend((f"train_{case}", train, case, True)
                         for case in ("orig", "q75", "moderate", "strong"))
        for key, rows, case, features in tasks:
            result = evaluate(model, rows, transform, case, device, args.batch_size, args.workers, features)
            if not key.startswith("test"):
                result["labels"] = [row["label"] for row in rows]
            with (args.output / f"{name}__{key}.npz").open("xb") as file:
                np.savez(file, **result)
            print(f"{name} {key} {len(rows)}", flush=True)


if __name__ == "__main__":
    main()

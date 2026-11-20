import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import PIL
import torch
import torchvision
from torchvision.models import mobilenet_v2, resnet18

from detection.data import (
    EpochSampler, ImageDataset, estimate_normalization, fingerprint, make_loader, read_split,
    read_test,
)
from detection.training import (
    atomic_file, evaluate_pair, fit, load_checkpoint, predict, seed_all, write_csv, write_json,
)


REPO = Path(__file__).resolve().parents[1]
MODELS = {"resnet18": resnet18, "mobilenet_v2": mobilenet_v2}
METRICS = ("accuracy", "auroc", "recall_real", "recall_ai")


def read_json(path):
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def build_model(name, device):
    memory_format = torch.channels_last if device.type == "xpu" else torch.contiguous_format
    return MODELS[name](weights=None, num_classes=2).to(device, memory_format=memory_format)


def read_config(args):
    config = read_json(args.config)
    if args.microbatch is not None:
        config["microbatch"] = args.microbatch
    microbatch = config["microbatch"]
    effective_batch = config["effective_batch"]
    if not 2 <= microbatch <= effective_batch or effective_batch % microbatch:
        raise ValueError("Microbatch must be at least two and divide the effective batch size")
    if not (0 < config["crop_size"] <= config["short_side"] and config["epochs"] >= 1):
        raise ValueError("Invalid image geometry or epoch count")
    if any(not 0 <= config[key] <= 1
           for key in ("flip_probability", "comparison_jpeg_probability")):
        raise ValueError("Augmentation probabilities must be in [0,1]")
    if not 0 < config["min_lr"] <= config["max_lr"]:
        raise ValueError("Learning rates must be positive and min_lr <= max_lr")
    if not 1 <= config["jpeg_quality"] <= 95 or config["jpeg_subsampling"] not in (0, 1, 2):
        raise ValueError("Invalid JPEG quality or subsampling")
    if (2 * config["train_per_class"]) % effective_batch:
        raise ValueError("Training sample count must be divisible by the effective batch size")
    seeds = [config["screen_seed"], *config["repeat_seeds"]]
    if len(config["repeat_seeds"]) != 3 or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        raise ValueError("Use one screening seed and three distinct non-negative repeat seeds")
    return config


def device_for(name, threads):
    if threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(threads)
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch environment")
    if name == "xpu" and not torch.xpu.is_available():
        raise ValueError("Intel XPU is unavailable in this PyTorch environment")
    return torch.device(name)


def data_identity(manifest, config):
    return {
        "split_sha256": fingerprint(manifest),
        "short_side": config["short_side"],
        "crop_size": config["crop_size"],
        "train_count": 2 * config["train_per_class"],
        "val_count": 2 * config["val_per_class"],
    }


def image_dataset(samples, config, *, norm=None, train=False, jpeg_probability=0.0):
    return ImageDataset(
        samples, config["short_side"], config["crop_size"], norm=norm, train=train,
        flip_probability=config["flip_probability"], jpeg_probability=jpeg_probability,
        quality=config["jpeg_quality"], subsampling=config["jpeg_subsampling"],
    )


def load_samples(args, config):
    groups = read_split(args.manifest, args.image_root, {
        "train": config["train_per_class"], "val": config["val_per_class"],
    })
    return groups, data_identity(args.manifest, config)


def ensure_normalization(path, train_samples, identity, config, workers):
    if path.exists():
        norm = read_json(path)
        if any(norm.get(key) != value for key, value in identity.items()):
            raise ValueError(f"{path} uses a different split or image geometry. Choose another path.")
        for key in ("mean", "std"):
            if len(norm[key]) != 3 or not all(map(math.isfinite, norm[key])):
                raise ValueError(f"{path} contains invalid {key} values")
        if any(value <= 0 for value in norm["std"]):
            raise ValueError(f"{path} contains non-positive standard deviations")
        return norm
    dataset = image_dataset(train_samples, config)
    loader = make_loader(dataset, config["microbatch"], workers)
    norm = {**identity, **estimate_normalization(loader)}
    write_json(path, norm)
    print(f"Saved training normalization to {path}", flush=True)
    return norm


def validation_loaders(samples, config, norm, workers, device, batch_size=None):
    datasets = [image_dataset(samples, config, norm=norm, jpeg_probability=prob)
                for prob in (0.0, 1.0)]
    return [make_loader(dataset, batch_size or config["microbatch"], workers,
                        pin_memory=device.type == "cuda", persistent=True) for dataset in datasets]


def code_version():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment(device, args):
    device_name = platform.processor() or platform.machine()
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "xpu":
        device_name = torch.xpu.get_device_name(device)
    return {
        "device": device.type, "device_name": device_name,
        "precision": "fp16_amp" if device.type in ("cuda", "xpu") else "fp32",
        "workers": args.workers, "threads": args.threads, "python": platform.python_version(),
        "torch": str(torch.__version__), "torchvision": torchvision.__version__,
        "pillow": PIL.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
        "git_commit": code_version(),
        "source_sha256": {
            path.relative_to(REPO).as_posix(): fingerprint(path)
            for path in [*sorted((REPO / "detection").glob("*.py")), Path(__file__).resolve()]
        },
    }


@contextmanager
def run_lock(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".lock").open("a+b") as file:
        file.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(f"An active training process already owns {run_dir}") from exc
        yield


def train_one(args, config, groups, norm, device, model_name, seed, jpeg_augmentation, run_dir):
    seed_all(seed, device)
    model = build_model(model_name, device)
    metadata = {
        "model": model_name,
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "seed": seed,
        "jpeg_train_probability": config["comparison_jpeg_probability"] if jpeg_augmentation else 0.0,
        "config": config,
        "norm": norm,
        "train_count": len(groups["train"]),
        "val_count": len(groups["val"]),
        "environment": environment(device, args),
    }
    train_dataset = image_dataset(
        groups["train"], config, norm=norm, train=True,
        jpeg_probability=metadata["jpeg_train_probability"],
    )
    sampler = EpochSampler(train_dataset)
    train_batches = make_loader(train_dataset, config["microbatch"], args.workers,
                                sampler=sampler, seed=seed, pin_memory=device.type == "cuda",
                                persistent=True)

    def train_loader(epoch_seed):
        sampler.seed = epoch_seed
        return train_batches

    val_loader, jpeg_loader = validation_loaders(groups["val"], config, norm, args.workers, device)
    with run_lock(run_dir):
        has_state = (run_dir / "last.pt").exists()
        if has_state and not args.resume:
            raise ValueError(f"{run_dir} already contains a run. Use --resume or a different output path.")
        if args.resume and (run_dir / "summary.json").exists() and (run_dir / "best.pt").is_file():
            summary = read_json(run_dir / "summary.json")
            if all(summary.get(key) == value for key, value in metadata.items()):
                return summary
        if not has_state:
            write_json(run_dir / "config.json", config)
        return fit(model, train_loader, val_loader, jpeg_loader, device, run_dir, metadata,
                   resume=has_state, epoch_limit=getattr(args, "epoch_limit", None))


def choose_winner(summaries):
    return max(summaries, key=lambda run: (run["validation"]["original"]["auroc"],
                                          run["validation"]["original"]["accuracy"],
                                          -run["params"]))


def write_comparison(output, winner, screens, repeats, control):
    with atomic_file(output / "comparison.md", "w") as file:
        file.write(f"# CNN 对照结果\n\n初筛胜出配置为 `{winner['model']}`。\n\n")
        file.write("混淆矩阵的行是真值，列是预测，类别顺序为真实、AI。"
                   "耗时为训练阶段每轮均值。\n\n")
        file.write("| 模型 | 种子 | JPEG 增强概率 | 验证图 | 准确率 | AUROC | 真实召回 | AI 召回 | "
                   "混淆矩阵 | 参数量 | 微批量 × 累积 | 秒/轮 | GPU 分配峰值 MiB | 设备 | 最佳权重 |\n"
                   "|---|---:|---:|---|---:|---:|---:|---:|---|---:|---|---:|---:|---|---|\n")
        for run in [*screens, *repeats, control]:
            memory = run["peak_allocated_bytes"]
            memory_text = "—" if memory is None else f"{memory / 2**20:.1f}"
            cfg = run["config"]
            accumulation = cfg["effective_batch"] // cfg["microbatch"]
            ckpt = Path(run["best_checkpoint"]).relative_to(output.resolve()).as_posix()
            for version in ("original", "jpeg"):
                values = run["validation"][version]
                label = "原图" if version == "original" else "JPEG"
                scores = " | ".join(f"{values[key]:.4f}" for key in METRICS)
                file.write(f"| {run['model']} | {run['seed']} | {run['jpeg_train_probability']:.2f} | "
                           f"{label} | {scores} | {values['confusion_matrix']} | {run['params']:,} | "
                           f"{cfg['microbatch']} × {accumulation} | {run['mean_train_seconds']:.2f} | "
                           f"{memory_text} | {run['environment']['device_name']} | "
                           f"[best.pt]({ckpt}) |\n")

        file.write("\n## 胜出配置三个种子的均值与样本标准差\n\n")
        file.write("| 验证图 | 准确率 | AUROC | 真实召回 | AI 召回 |\n"
                   "|---|---:|---:|---:|---:|\n")
        for version, label in (("original", "原图"), ("jpeg", "JPEG")):
            columns = []
            for key in METRICS:
                values = [run["validation"][version][key] for run in repeats]
                columns.append(f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}")
            file.write(f"| {label} | " + " | ".join(columns) + " |\n")

        file.write("\n## JPEG 重编码与增强的差值\n\n")
        file.write("| 模型与种子 | 对照 | Δ 准确率 | Δ AUROC | Δ 真实召回 | Δ AI 召回 |\n"
                   "|---|---|---:|---:|---:|---:|\n")
        for run in [*screens, control]:
            deltas = run["validation"]["delta_jpeg_minus_original"]
            columns = " | ".join(f"{deltas[key]:+.4f}" for key in METRICS)
            label = "JPEG − 原图（增强）" if run is control else "JPEG − 原图"
            file.write(f"| {run['model']} / {run['seed']} | {label} | {columns} |\n")
        for version, label in (("original", "原图"), ("jpeg", "JPEG")):
            columns = " | ".join(
                f"{control['validation'][version][key] - winner['validation'][version][key]:+.4f}"
                for key in METRICS
            )
            file.write(f"| {winner['model']} / {winner['seed']} | {label} · 增强 − 基础 | {columns} |\n")


def run_suite(args, config, groups, norm, device):
    seed = config["screen_seed"]
    with run_lock(args.output):
        screens = [
            train_one(args, config, groups, norm, device, model, seed, False,
                      args.output / f"{model}-s{seed}")
            for model in MODELS
        ]
        winner = choose_winner(screens)
        name = winner["model"]
        repeats = [
            train_one(args, config, groups, norm, device, name, repeat_seed, False,
                      args.output / f"{name}-s{repeat_seed}")
            for repeat_seed in config["repeat_seeds"]
        ]
        control = train_one(args, config, groups, norm, device, name, seed, True,
                            args.output / f"{name}-s{seed}-jpeg")
        write_comparison(args.output, winner, screens, repeats, control)
        write_json(args.output / "selection.json", {
            "model": name, "seed": seed,
            "checkpoint": f"{name}-s{seed}/best.pt",
        })
        print(f"Saved {args.output / 'comparison.md'}", flush=True)


def run_checkpoint(args, device):
    checkpoint = load_checkpoint(args.checkpoint)
    metadata = checkpoint["metadata"]
    config, norm = metadata["config"], metadata["norm"]
    model = build_model(metadata["model"], device)
    model.load_state_dict(checkpoint["model_state"])
    if args.command == "evaluate":
        groups, identity = load_samples(args, config)
        if any(norm[key] != value for key, value in identity.items()):
            raise ValueError("Validation split or geometry differs from checkpoint")
        original, jpeg = validation_loaders(groups["val"], config, norm, args.workers, device,
                                            args.batch_size)
        metrics = evaluate_pair(model, original, jpeg, device, args.output)
        write_json(args.output / "validation.json", metrics)
    else:
        samples = read_test(args.template, args.test_dir)
        if len(samples) != config["test_count"]:
            raise ValueError(f"Expected {config['test_count']} test images, found {len(samples)}")
        dataset = image_dataset(samples, config, norm=norm)
        loader = make_loader(dataset, args.batch_size, args.workers, pin_memory=device.type == "cuda")
        predictions = predict(model, loader, device)
        if predictions["ids"] != [sample.image_id for sample in samples]:
            raise ValueError("Inference rows must follow the template order")
        write_csv(args.output, ("image_id", "label"),
                  zip(predictions["ids"], predictions["classes"], strict=True))
    print(f"Saved {args.output}", flush=True)


def parser():
    root = argparse.ArgumentParser(description="Train, evaluate and export classic CNN baselines.")
    root.add_argument("--device", choices=("auto", "cpu", "cuda", "xpu"), default="auto")
    root.add_argument("--workers", type=int, default=2)
    root.add_argument("--threads", type=int, default=4)
    commands = root.add_subparsers(dest="command", required=True)

    for name in ("prepare", "train", "suite", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--image-root", type=Path, default=Path("dataset"))
        command.add_argument("--manifest", type=Path, default=Path("splits/train_val.csv"))
        if name != "evaluate":
            command.add_argument("--config", type=Path, default=REPO / "lqy" / "config.json")
            command.add_argument("--norm", type=Path, default=Path("lqy/runs/normalization.json"))
            command.add_argument("--microbatch", type=int)

    train, suite = commands.choices["train"], commands.choices["suite"]
    for command in (train, suite):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--resume", action="store_true")
    train.add_argument("--model", choices=tuple(MODELS), required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--jpeg-augmentation", action="store_true")
    train.add_argument("--epoch-limit", type=int, help="Pause after this many epochs, keeping full budget.")

    evaluate = commands.choices["evaluate"]
    export = commands.add_parser("export")
    for command in (evaluate, export):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--batch-size", type=int, default=32)
    export.add_argument("--template", type=Path,
                        default=Path("ai-generated-image-detection-benchmark/sample_submission.csv"))
    export.add_argument("--test-dir", type=Path, default=Path("dataset/dataset/test"))
    return root


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        if args.workers < 0:
            raise ValueError("--workers cannot be negative")
        if getattr(args, "batch_size", 1) < 1:
            raise ValueError("--batch-size must be positive")
        if getattr(args, "epoch_limit", None) is not None and args.epoch_limit < 1:
            raise ValueError("--epoch-limit must be positive")
        device = device_for(args.device, args.threads)
        if args.command in ("evaluate", "export"):
            run_checkpoint(args, device)
            return
        config = read_config(args)
        groups, identity = load_samples(args, config)
        norm = ensure_normalization(args.norm, groups["train"], identity, config, args.workers)
        if args.command == "train":
            seed = config["screen_seed"] if args.seed is None else args.seed
            if seed < 0:
                raise ValueError("--seed cannot be negative")
            train_one(args, config, groups, norm, device, args.model, seed,
                      args.jpeg_augmentation, args.output)
        elif args.command == "suite":
            run_suite(args, config, groups, norm, device)
    except (ValueError, FileNotFoundError) as exc:
        cli.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

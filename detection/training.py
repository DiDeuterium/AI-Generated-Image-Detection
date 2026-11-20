import csv
import json
import math
import os
import random
import tempfile
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from .metrics import binary_metrics


@contextmanager
def atomic_file(path, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {} if "b" in mode else {"encoding": "utf-8", "newline": ""}
    with tempfile.NamedTemporaryFile(mode, dir=path.parent, delete=False, **options) as file:
        temp = file.name
        try:
            yield file
            file.close()
            os.replace(temp, path)
        finally:
            file.close()
            if os.path.exists(temp):
                os.unlink(temp)


def write_json(path, payload):
    with atomic_file(path, "w") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def write_csv(path, columns, rows):
    with atomic_file(path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def save_checkpoint(path, payload):
    with atomic_file(path, "wb") as file:
        torch.save(payload, file)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)


def move_images(images, device):
    memory_format = torch.channels_last if device.type == "xpu" else torch.contiguous_format
    return images.to(device, non_blocking=True, memory_format=memory_format)


def learning_rate(update, updates, warmup, min_lr, max_lr):
    # Zero-based optimizer update. Include both endpoints in the warmup.
    if update < warmup:
        return min_lr + (max_lr - min_lr) * update / max(1, warmup - 1)
    progress = (update + 1 - warmup) / (updates - warmup)
    return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


def train_epoch(model, loader, optimizer, scaler, device, config, step, total_steps, ema=None):
    model.train()
    accumulation = config["effective_batch"] // config["microbatch"]
    if len(loader) % accumulation or len(loader.dataset) % config["effective_batch"]:
        raise ValueError("Training sample count must be divisible by the effective batch size")
    criterion = nn.CrossEntropyLoss()
    loss_sum = torch.zeros((), device=device)
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    accelerator = {"cuda": torch.cuda, "xpu": torch.xpu}.get(device.type)
    if accelerator:
        accelerator.reset_peak_memory_stats(device)
    start = time.perf_counter()
    skipped = 0
    for index, (images, targets, _) in enumerate(loader, start=1):
        images = move_images(images, device)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            loss = criterion(model(images), targets)
        loss_sum += loss.detach() * targets.numel()
        scaler.scale(loss / accumulation).backward()
        if index % accumulation:
            continue
        lr = learning_rate(step, total_steps, config["warmup_updates"],
                           config["min_lr"], config["max_lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() < previous_scale:
            skipped += 1
        else:
            step += 1
            if ema is not None:
                ema.update_parameters(model)
    synchronize(device)
    seconds = time.perf_counter() - start
    mean_loss = loss_sum.item() / len(loader.dataset)
    if not math.isfinite(mean_loss):
        raise ValueError("Non-finite training loss")
    memory = accelerator.max_memory_allocated(device) if accelerator else None
    return {
        "train_loss": mean_loss,
        "train_seconds": seconds,
        "peak_allocated_bytes": memory,
        "optimizer_updates": step,
        "skipped_updates": skipped,
        "last_lr": lr,
    }


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    ids, labels, classes, scores = [], [], [], []
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = torch.zeros((), device=device)
    for images, targets, batch_ids in loader:
        logits = model(move_images(images, device)).float()
        if not torch.isfinite(logits).all():
            raise ValueError(f"Non-finite logits for batch starting at {batch_ids[0]}")
        classes.extend(logits.argmax(dim=1).cpu().tolist())
        scores.extend(logits.softmax(dim=1)[:, 1].cpu().tolist())
        ids.extend(batch_ids)
        labels.extend(targets.tolist())
        if targets[0] >= 0:
            loss_sum += criterion(logits, targets.to(device, non_blocking=True))
    return {
        "ids": ids, "labels": labels, "classes": classes, "scores": scores,
        "loss": loss_sum.item() / len(ids),
    }


def evaluate(model, loader, device):
    predictions = predict(model, loader, device)
    metrics = binary_metrics(predictions["labels"], predictions["classes"], predictions["scores"])
    metrics["loss"] = predictions["loss"]
    return metrics, predictions


def evaluate_pair(model, original_loader, jpeg_loader, device, output_dir):
    original, original_predictions = evaluate(model, original_loader, device)
    jpeg, jpeg_predictions = evaluate(model, jpeg_loader, device)
    ids = original_predictions["ids"]
    labels = original_predictions["labels"]
    if ids != jpeg_predictions["ids"] or labels != jpeg_predictions["labels"]:
        raise ValueError("Original and JPEG validation rows are misaligned")
    write_csv(
        output_dir / "val_predictions.csv",
        ("image_id", "label", "original_class", "original_score", "jpeg_class", "jpeg_score"),
        zip(ids, labels, original_predictions["classes"], original_predictions["scores"],
            jpeg_predictions["classes"], jpeg_predictions["scores"], strict=True),
    )
    return {
        "original": original,
        "jpeg": jpeg,
        "delta_jpeg_minus_original": {
            key: jpeg[key] - original[key]
            for key in ("accuracy", "auroc", "recall_real", "recall_ai")
        },
    }


def _write_history(path, history):
    rows = []
    for entry in history:
        metrics = entry["original"]
        row = {key: value for key, value in entry.items() if key != "original"}
        row.update({f"val_{key}": value for key, value in metrics.items() if key != "confusion_matrix"})
        row.update(zip(("tn", "fp", "fn", "tp"), sum(metrics["confusion_matrix"], []), strict=True))
        rows.append(row)
    fields = tuple(rows[0])
    write_csv(path, fields, ([row[field] for field in fields] for row in rows))


def fit(model, train_loader, val_loader, jpeg_loader, device, run_dir, metadata, *, resume=False,
        epoch_limit=None):
    """Train a binary model. train_loader(epoch_seed) returns the shuffled epoch loader."""
    config = metadata["config"]
    seed = metadata["seed"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["min_lr"], betas=tuple(config["betas"]),
        eps=config["eps"], weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler(device.type, init_scale=4096,
                                  enabled=device.type in ("cuda", "xpu"))
    decay = config.get("ema_decay", 0.0)
    ema = None
    if decay:
        ema = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay),
            use_buffers=True,
        )
    eval_model = model if ema is None else ema.module
    epochs = config["epochs"]
    steps_per_epoch = metadata["train_count"] // config["effective_batch"]
    total_steps = epochs * steps_per_epoch
    if not 0 <= config["warmup_updates"] < total_steps:
        raise ValueError("Warmup must be shorter than the total training budget")
    state_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    history, best = [], None
    if resume:
        state = load_checkpoint(state_path)
        if state["metadata"] != metadata:
            raise ValueError(f"{state_path} uses different settings, data, code or environment")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        if ema is not None:
            ema.load_state_dict(state["ema"])
        history, best = state["history"], state["best"]

    step = history[-1]["optimizer_updates"] if history else 0
    stop_epoch = epochs if epoch_limit is None else min(epochs, epoch_limit)
    for epoch in range(len(history) + 1, stop_epoch + 1):
        epoch_seed = int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0])
        seed_all(epoch_seed, device)
        entry = train_epoch(model, train_loader(epoch_seed), optimizer, scaler,
                            device, config, step, total_steps, ema)
        step = entry["optimizer_updates"]
        start = time.perf_counter()
        metrics, _ = evaluate(eval_model, val_loader, device)
        entry.update(epoch=epoch, original=metrics, val_seconds=time.perf_counter() - start)
        history.append(entry)

        if best is None or metrics["auroc"] > best["original_auroc"]:
            best = {
                "epoch": epoch,
                "original_auroc": metrics["auroc"],
                "model_state": {key: value.detach().to("cpu", copy=True)
                                for key, value in eval_model.state_dict().items()},
            }
        state = {
            "metadata": metadata, "history": history, "best": best,
            "model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }
        if ema is not None:
            state["ema"] = ema.state_dict()
        save_checkpoint(state_path, state)
        _write_history(run_dir / "epochs.csv", history)
        print(f"{run_dir.name} epoch {epoch}/{epochs}  loss={entry['train_loss']:.4f} "
              f"val_AUROC={metrics['auroc']:.4f}  train={entry['train_seconds']:.1f}s",
              flush=True)

    # Rebuild exported artifacts from the checkpoint if interrupted after checkpointing.
    _write_history(run_dir / "epochs.csv", history)
    if len(history) < epochs:
        return None
    save_checkpoint(best_path, {"metadata": metadata, **best})
    checkpoint = load_checkpoint(best_path)
    eval_model.load_state_dict(checkpoint["model_state"])
    validation = evaluate_pair(eval_model, val_loader, jpeg_loader, device, run_dir)
    peaks = [entry["peak_allocated_bytes"] for entry in history
             if entry["peak_allocated_bytes"] is not None]
    summary = {
        **metadata,
        "best_epoch": checkpoint["epoch"],
        "best_checkpoint": str(best_path.resolve()),
        "validation": validation,
        "optimizer_updates": step,
        "skipped_updates": sum(entry["skipped_updates"] for entry in history),
        "mean_train_seconds": sum(entry["train_seconds"] for entry in history) / len(history),
        "peak_allocated_bytes": max(peaks) if peaks else None,
    }
    write_json(run_dir / "summary.json", summary)
    return summary

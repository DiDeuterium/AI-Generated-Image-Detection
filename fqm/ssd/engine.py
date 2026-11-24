"""训练循环、验证与指标计算（统一种子、AMP、按更新计数调度学习率）。"""

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from .view_consistency import paired_view_loss


# --------------------------------------------------------------------------------------
# 随机性
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_amp(amp: bool, dtype_name: str, device: torch.device):
    """返回 (是否启用 autocast, dtype)；CPU 上关闭混合精度。"""
    if not amp or device.type != 'cuda':
        return False, torch.float32
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16}[dtype_name]
    return True, dtype


# --------------------------------------------------------------------------------------
# 学习率：前 warmup 步线性升温，其余余弦衰减到 lr_final
# --------------------------------------------------------------------------------------
def lr_at_step(step: int, base_lr: float, start_lr: float, final_lr: float,
               warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        ratio = (step + 1) / max(1, warmup_steps)
        return start_lr + (base_lr - start_lr) * ratio
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return final_lr + 0.5 * (base_lr - final_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: nn.Module, lr: float, weight_decay: float,
                    betas: Tuple[float, float], eps: float) -> torch.optim.Optimizer:
    """AdamW，全体可训练参数统一 weight decay（按统一协议，不做参数分组排除）。"""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


# --------------------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------------------
def best_threshold_accuracy(labels: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
    """在给定验证集上扫描阈值，返回 (最高准确率, 阈值)；仅供参考，不作为主指标。"""
    scores = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    num_pos = float(labels.sum())
    num_neg = float(len(labels) - num_pos)
    accuracy = (tpr * num_pos + (1.0 - fpr) * num_neg) / float(len(labels))
    index = int(np.argmax(accuracy))
    return float(accuracy[index]), float(thresholds[index])


def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, object]:
    """正类为 1（AI 生成）。返回准确率、AUROC、两类召回率与混淆矩阵（行真值、列预测）。"""
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    recall0 = float(cm[0, 0] / cm[0].sum()) if cm[0].sum() > 0 else float('nan')
    recall1 = float(cm[1, 1] / cm[1].sum()) if cm[1].sum() > 0 else float('nan')
    auroc = float(roc_auc_score(labels, probs[:, 1])) if len(np.unique(labels)) > 1 else float('nan')
    return {
        'accuracy': float((preds == labels).mean()),
        'auroc': auroc,
        'recall_real': recall0,
        'recall_fake': recall1,
        'confusion_matrix': cm.astype(int).tolist(),
        'num_samples': int(len(labels)),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device,
             amp: bool, amp_dtype: torch.dtype) -> Dict[str, object]:
    """在验证集上前向，返回损失、标签与 softmax 概率。"""
    model.eval()
    labels_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    loss_sum, count = 0.0, 0
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)
        loss_sum += float(loss) * labels.numel()
        count += labels.numel()
        labels_all.append(labels.cpu().numpy())
        probs_all.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    labels_np = np.concatenate(labels_all)
    probs_np = np.concatenate(probs_all)
    metrics = compute_metrics(labels_np, probs_np)
    metrics.update({'loss': loss_sum / max(1, count), 'eval_seconds': elapsed})
    return {'metrics': metrics, 'labels': labels_np, 'probs': probs_np}


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device, amp: bool,
            amp_dtype: torch.dtype) -> np.ndarray:
    """纯推理：返回按样本顺序排列的 softmax 概率（用于测试集导出）。"""
    model.eval()
    probs_all: List[np.ndarray] = []
    for images, _, _ in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            logits = model(images)
        probs_all.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(probs_all) if probs_all else np.zeros((0, 2))


# --------------------------------------------------------------------------------------
# 损失：支持硬标签、软标签（mixup / cutmix）与 focal loss
# --------------------------------------------------------------------------------------
class SoftTargetCrossEntropy(nn.Module):
    """软标签交叉熵；标签可以是一维硬标签或 (B, C) 软目标。"""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        if target.dim() == 1:
            target = torch.nn.functional.one_hot(target, num_classes).to(logits.dtype)
        if self.label_smoothing > 0:
            target = target * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        return -(target.float() * log_probs).sum(dim=-1).mean()


class FocalLoss(nn.Module):
    """Focal loss（硬标签）：通过 (1-p_t)^gamma 降低易分样本权重。"""

    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        log_pt = log_probs.gather(1, target.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        return (-(1.0 - pt) ** self.gamma * log_pt).mean()


def build_criterion(train_cfg: Dict) -> nn.Module:
    """按配置构造损失函数；mixup/cutmix 时自动改为软标签交叉熵。"""
    loss_kind = str(train_cfg.get('loss', 'ce')).lower()
    mixing = float(train_cfg.get('mixup_alpha', 0.0) or 0.0) > 0 or \
        float(train_cfg.get('cutmix_alpha', 0.0) or 0.0) > 0
    smoothing = float(train_cfg.get('label_smoothing', 0.0))
    if mixing:
        return SoftTargetCrossEntropy(label_smoothing=smoothing)
    if loss_kind == 'focal':
        return FocalLoss(gamma=float(train_cfg.get('focal_gamma', 1.0)))
    if loss_kind == 'ce' and smoothing > 0:
        return SoftTargetCrossEntropy(label_smoothing=smoothing)
    return nn.CrossEntropyLoss(label_smoothing=smoothing)


def mixup_batch(images: torch.Tensor, targets: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, targets, targets[index], lam


def cutmix_batch(images: torch.Tensor, targets: torch.Tensor, alpha: float):
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    index = torch.randperm(images.size(0), device=images.device)
    height, width = images.shape[2], images.shape[3]
    ratio = float(np.sqrt(1.0 - lam))
    cut_w, cut_h = int(width * ratio), int(height * ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1, x2 = max(0, cx - cut_w // 2), min(width, cx + cut_w // 2)
    y1, y2 = max(0, cy - cut_h // 2), min(height, cy + cut_h // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index][:, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    return mixed, targets, targets[index], lam


def soft_targets(target_a: torch.Tensor, target_b: torch.Tensor, lam: float,
                 num_classes: int) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(target_a, num_classes).float()
    other = torch.nn.functional.one_hot(target_b, num_classes).float()
    return lam * one_hot + (1.0 - lam) * other


# --------------------------------------------------------------------------------------
# 权重平均（EMA）
# --------------------------------------------------------------------------------------
class ModelEma:
    """训练权重的指数滑动平均副本，只用于评测与保存（不参与反向）。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = float(decay)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for ema_param, param in zip(self.module.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        for ema_buffer, buffer in zip(self.module.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def train_one_epoch(model: nn.Module, loader, optimizer, scaler, criterion,
                    device: torch.device, amp: bool, amp_dtype: torch.dtype,
                    accum_steps: int, updates_per_epoch: int, total_updates: int,
                    base_lr: float, start_lr: float, final_lr: float,
                    warmup_updates: int, global_update: int,
                    mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0,
                    mix_prob: float = 1.0, ema: Optional[ModelEma] = None,
                    num_classes: int = 2, consistency_weight: float = 0.0) -> Dict[str, float]:
    """训练到 `updates_per_epoch` 次参数更新；返回损失、准确率、用时与耗时统计。

    `global_update` 是本 epoch 开始前已完成的优化器更新次数，用于跨 epoch 连续调度学习率。
    mixup/cutmix 打开时按批随机施加并改用软目标损失。
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, correct, seen = 0.0, 0, 0
    micro_in_group = 0
    updates = 0
    last_lr = float('nan')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        target = labels
        if (mixup_alpha > 0 or cutmix_alpha > 0) and float(np.random.rand()) < mix_prob:
            if mixup_alpha > 0 and (cutmix_alpha <= 0 or np.random.rand() < 0.5):
                images, target_a, target_b, lam = mixup_batch(images, labels, mixup_alpha)
            else:
                images, target_a, target_b, lam = cutmix_batch(images, labels, cutmix_alpha)
            target = soft_targets(target_a, target_b, lam, num_classes)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            if images.ndim == 5:
                pair_logits = model(images.flatten(0, 1)).reshape(labels.numel(), 2, -1)
                loss = paired_view_loss(pair_logits, labels, criterion, consistency_weight)
                logits = pair_logits[:, 1]
            else:
                logits = model(images)
                loss = criterion(logits, target)
            loss = loss / accum_steps
        scaler.scale(loss).backward()

        micro_in_group += 1
        loss_sum += float(loss.detach()) * accum_steps * labels.numel()
        correct += int((logits.detach().argmax(dim=1) == labels).sum())
        seen += labels.numel()
        if micro_in_group < accum_steps:
            continue

        step = global_update + updates
        last_lr = lr_at_step(step, base_lr, start_lr, final_lr, warmup_updates,
                             total_updates)
        for group in optimizer.param_groups:
            group['lr'] = last_lr
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model)
        micro_in_group = 0
        updates += 1
        if updates >= updates_per_epoch:
            break
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return {
        'loss': loss_sum / max(1, seen),
        'accuracy': correct / max(1, seen),
        'seconds': elapsed,
        'updates': updates,
        'images': seen,
        'lr': last_lr,
        'peak_memory_mb': (torch.cuda.max_memory_allocated() / 1024 ** 2
                           if device.type == 'cuda' else 0.0),
    }


# --------------------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------------------
def save_checkpoint(path: Path, model: nn.Module, optimizer, epoch: int,
                    metrics: Dict[str, object], config: Dict,
                    extra: Optional[Dict] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics,
        'config': config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device,
                    optimizer=None) -> Dict:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload['model'])
    if optimizer is not None and 'optimizer' in payload:
        optimizer.load_state_dict(payload['optimizer'])
    return payload


def append_jsonl(path: Path, record: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

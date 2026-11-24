"""开训前自检：一次真实批次的前向、反向与参数更新，一次验证，以及 checkpoint 复现。

对应 fqm/需求文档.md「实验顺序」第 2 条与 doc/项目与数据说明.md 的「环境与开训检查」。
用法（仓库根目录）：
    python fqm/smoke_test.py --config fqm/configs/efficientvim_m1.json --batch-size 8
"""

import argparse
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssd.data import (build_loader, build_transform, compute_norm_stats,
                      load_norm_stats, read_splits)
from ssd.efficientvim import efficientvim_m1
from ssd.engine import (build_optimizer, evaluate, load_checkpoint, resolve_amp,
                        save_checkpoint, set_seed, train_one_epoch)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='EfficientViM 开训自检')
    parser.add_argument('--config', type=Path, default=Path('fqm/configs/efficientvim_m1.json'))
    parser.add_argument('--batch-size', type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    data_cfg, train_cfg = cfg['data'], cfg['train']
    seed = int(train_cfg['seed'])
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[env] python {platform.python_version()} | torch {torch.__version__} | '
          f'cuda {torch.version.cuda} | cudnn {torch.backends.cudnn.version()} | '
          f'device {device}')
    if device.type == 'cuda':
        print(f'[env] gpu {torch.cuda.get_device_name(device)} | '
              f'{torch.cuda.get_device_properties(device).total_memory / 1024 ** 3:.1f} GiB')

    train_rows, val_rows = read_splits(REPO_ROOT / data_cfg['splits'],
                                       REPO_ROOT / data_cfg['image_dir'])
    batch_size = args.batch_size
    train_rows = train_rows[:batch_size * 2]
    val_rows = val_rows[:batch_size]

    stats = load_norm_stats(REPO_ROOT / data_cfg['norm_stats'])
    if stats is None:
        print('[data] 尚未缓存归一化统计，用本次小样本临时计算（不写盘）')
        stats = compute_norm_stats(train_rows, data_cfg['input_size'],
                                   data_cfg['resize_size'], num_workers=4)
    mean, std = stats['mean'], stats['std']
    train_tf = build_transform(data_cfg['input_size'], data_cfg['resize_size'],
                               mean, std, train=True)
    val_tf = build_transform(data_cfg['input_size'], data_cfg['resize_size'],
                             mean, std, train=False)
    train_loader = build_loader(train_rows, train_tf, batch_size, shuffle=True,
                                num_workers=0, seed=seed, drop_last=True)
    val_loader = build_loader(val_rows, val_tf, batch_size, shuffle=False,
                              num_workers=0, seed=seed)

    model = efficientvim_m1(num_classes=int(cfg['model']['num_classes'])).to(device)
    param_total = sum(p.numel() for p in model.parameters())
    print(f'[model] {cfg["model"]["name"]} 参数量 {param_total / 1e6:.3f}M，随机初始化')

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, float(train_cfg['lr']),
                                float(train_cfg['weight_decay']),
                                tuple(train_cfg['betas']), float(train_cfg['eps']))
    amp, amp_dtype = resolve_amp(bool(train_cfg['amp']), train_cfg['amp_dtype'], device)
    scaler = GradScaler(device.type, enabled=amp)

    before = model.patch_embed.conv[0].conv.weight.detach().clone()
    print('[check] 前向 + 反向 + 一次参数更新 ...')
    start = time.time()
    log = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device,
                          amp, amp_dtype, accum_steps=1, updates_per_epoch=1,
                          total_updates=1, base_lr=float(train_cfg['lr']),
                          start_lr=float(train_cfg['lr_start']),
                          final_lr=float(train_cfg['lr_final']),
                          warmup_updates=1, global_update=0)
    after = model.patch_embed.conv[0].conv.weight.detach().clone()
    changed = not torch.equal(before, after)
    print(f'[check] 更新完成 {time.time() - start:.1f}s | loss {log["loss"]:.4f} | '
          f'权重已更新={changed} | peak {log["peak_memory_mb"]:.0f}MB')
    if not changed:
        print('[smoke] FAIL: 参数没有更新')
        return 1

    print('[check] 跑一次验证 ...')
    val_out = evaluate(model, val_loader, criterion, device, amp, amp_dtype)
    print(f'[check] val loss {val_out["metrics"]["loss"]:.4f} '
          f'acc {val_out["metrics"]["accuracy"]:.4f}')

    print('[check] checkpoint 保存/重载复现 ...')
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = Path(tmp) / 'smoke.pt'
        save_checkpoint(ckpt, model, optimizer, 1, val_out['metrics'], cfg)
        fresh = efficientvim_m1(num_classes=int(cfg['model']['num_classes'])).to(device)
        load_checkpoint(ckpt, fresh, device)
        model.eval()
        fresh.eval()
        with torch.no_grad():
            images, _, _ = next(iter(val_loader))
            images = images.to(device)
            diff = (model(images).float() - fresh(images).float()).abs().max().item()
    print(f'[check] 重载后 logits 最大差异 {diff:.2e}')
    if diff > 1e-5:
        print('[smoke] FAIL: checkpoint 复现失败')
        return 1

    print('[smoke] PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())

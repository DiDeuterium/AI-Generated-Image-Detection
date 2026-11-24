#!/usr/bin/env python
"""EfficientViM-M1 二分类基线：从随机初始化训练，不使用任何预训练权重。

按仓库「统一实验协议」与 ddc/实验设置.md 的公共设置：
6400/1600 固定划分、256->224 统一预处理、AdamW、有效 batch 64、30 epoch × 100 次更新、
FP16 AMP、种子 42。验证用原图 AUROC 选模，最后用同一权重评测统一 JPEG 重编码版。

用法（仓库根目录）：
    python fqm/train.py --config fqm/configs/efficientvim_m1.json
"""

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssd.data import (build_loader, build_transform, compute_norm_stats,
                      ensure_jpeg_cache, load_norm_stats, make_worker_init_fn, read_splits,
                      save_norm_stats)
from ssd.efficientvim import efficientvim_m1
from ssd.data import ImageDataset
from ssd.domain_shift import FixedDomainShift, PROXY_CASES, legacy_degrade
from ssd.target_profile import FixedTargetMatch, RandomTargetMatch, validation_groups
from ssd.target_training import PairedTargetSampler, PairedTrainingImages
from ssd.view_consistency import PairedDegradation
from torch.utils.data import DataLoader

from ssd.engine import (ModelEma, append_jsonl, best_threshold_accuracy, build_criterion,
                        build_optimizer, evaluate, load_checkpoint, resolve_amp,
                        save_checkpoint, set_seed, train_one_epoch)

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='训练 EfficientViM-M1 二分类基线')
    parser.add_argument('--config', type=Path, required=True, help='JSON 配置文件')
    parser.add_argument('--seed', type=int, default=None, help='覆盖配置中的随机种子')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖训练轮数')
    parser.add_argument('--micro-batch', type=int, default=None, help='覆盖微批量')
    parser.add_argument('--accum-steps', type=int, default=None, help='覆盖梯度累积次数')
    parser.add_argument('--updates-per-epoch', type=int, default=None,
                        help='覆盖每轮参数更新次数')
    parser.add_argument('--num-workers', type=int, default=None, help='覆盖 DataLoader worker 数')
    parser.add_argument('--run-name', type=str, default=None, help='覆盖运行目录名')
    parser.add_argument('--output-dir', type=Path, default=None, help='覆盖输出根目录')
    parser.add_argument('--limit-train', type=int, default=None,
                        help='仅用前 N 张训练图（调试用，正式结果不要使用）')
    parser.add_argument('--limit-val', type=int, default=None,
                        help='仅用前 N 张验证图（调试用，正式结果不要使用）')
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def apply_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    train_cfg, data_cfg = cfg['train'], cfg['data']
    if args.seed is not None:
        train_cfg['seed'] = args.seed
    if args.epochs is not None:
        train_cfg['epochs'] = args.epochs
    if args.micro_batch is not None:
        train_cfg['micro_batch'] = args.micro_batch
    if args.accum_steps is not None:
        train_cfg['accum_steps'] = args.accum_steps
    if args.updates_per_epoch is not None:
        train_cfg['updates_per_epoch'] = args.updates_per_epoch
    if args.num_workers is not None:
        data_cfg['num_workers'] = args.num_workers
    if args.output_dir is not None:
        cfg['output']['root'] = str(args.output_dir)
    if args.run_name is not None:
        cfg['output']['run_name'] = args.run_name
    return cfg


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def build_model(model_cfg: Dict):
    """按配置构造 EfficientViM；可选地把压缩伪迹先验拼到输入通道上。"""
    overrides = {key: model_cfg[key] for key in
                 ('embed_dim', 'depths', 'state_dim', 'mlp_ratio', 'ssd_expand',
                  'drop_rate', 'drop_path_rate') if key in model_cfg}
    num_classes = int(model_cfg['num_classes'])
    prior_cfg = model_cfg.get('prior')
    if not prior_cfg:
        return efficientvim_m1(num_classes=num_classes, **overrides)

    from ssd.prior import CompressionPrior, PriorAugmentedBackbone
    prior = CompressionPrior(parts=tuple(prior_cfg.get("parts", ("block", "highfreq"))),
                                 block=int(prior_cfg.get("block", 8)),
                                smooth=int(prior_cfg.get("smooth", 8)),
                                detach_feature=bool(prior_cfg.get('detach', False)))
    backbone = efficientvim_m1(num_classes=num_classes,
                               in_chans=3 + prior.out_channels, **overrides)
    return PriorAugmentedBackbone(backbone, prior)


def git_commit() -> str:
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return 'unknown'


def environment_info(device: torch.device) -> Dict[str, object]:
    info = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda,
        'device': str(device),
        'git_commit': git_commit(),
    }
    if device.type == 'cuda':
        info['gpu'] = torch.cuda.get_device_name(device)
        info['gpu_memory_mb'] = round(
            torch.cuda.get_device_properties(device).total_memory / 1024 ** 2, 1)
    return info


def prepare_norm_stats(cfg: Dict, train_rows, output_path: Path,
                       cache: bool = True) -> Dict:
    """归一化统计只由训练子集估计，首次运行时计算并缓存。"""
    stats = load_norm_stats(output_path)
    if stats is not None:
        print(f'[data] 复用归一化统计 {output_path}: mean={stats["mean"]} std={stats["std"]}')
        return stats
    print('[data] 未找到归一化统计，按验证几何变换在训练子集上计算...')
    start = time.time()
    stats = compute_norm_stats(train_rows, cfg['data']['input_size'],
                               cfg['data']['resize_size'], cfg['data']['num_workers'])
    if cache:
        save_norm_stats(stats, output_path)
    print(f'[data] 统计完成（{time.time() - start:.1f}s）: mean={stats["mean"]} '
          f'std={stats["std"]}')
    return stats


def legacy_strong(image, image_id):
    return legacy_degrade(image, image_id, 'strong')


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    train_cfg, data_cfg, model_cfg = cfg['train'], cfg['data'], cfg['model']

    seed = int(train_cfg['seed'])
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = environment_info(device)

    run_dir = repo_path(cfg['output']['root']) / cfg['output']['run_name']
    run_dir.mkdir(parents=True, exist_ok=not train_cfg.get('fresh_output', False))
    log_path = run_dir / 'log.jsonl'
    if log_path.exists():
        log_path.unlink()
    print(f'[run] 输出目录 {run_dir}')
    print(f'[env] {env}')

    # ------------------------------------------------------------------ 数据
    train_rows, val_rows = read_splits(repo_path(data_cfg['splits']),
                                       repo_path(data_cfg['image_dir']))
    if args.limit_train:
        train_rows = train_rows[:args.limit_train]
    if args.limit_val:
        val_rows = val_rows[:args.limit_val]
    target_settings = data_cfg.get('target_alignment')
    profiles = None
    select_rows = val_rows
    if target_settings:
        profiles = json.loads(repo_path(target_settings['profiles']).read_text())
        select_rows, _ = validation_groups(val_rows)
    print(f'[data] train={len(train_rows)} val={len(select_rows)} '
          f'(跳过测试集，仅用于最终推理)')

    norm_path = repo_path(data_cfg['norm_stats'])
    stats = prepare_norm_stats(cfg, train_rows, norm_path, cache=not args.limit_train)
    mean, std = stats['mean'], stats['std']

    aug_cfg = data_cfg.get('aug', {})
    train_tf = build_transform(data_cfg['input_size'], data_cfg['resize_size'],
                               mean, std, train=True, aug=aug_cfg)
    val_tf = build_transform(data_cfg['input_size'], data_cfg['resize_size'],
                             mean, std, train=False, aug=aug_cfg)
    if target_settings and target_settings['match_train']:
        if aug_cfg.get('domain_shift') or aug_cfg.get('jpeg_aug'):
            raise ValueError('Target matching replaces the previous degradation augmentation')
        train_tf.transforms.insert(0, RandomTargetMatch(profiles))
    print(f'[data] 增强配置 {aug_cfg}, target_alignment={target_settings}')

    micro_batch = int(train_cfg['micro_batch'])
    accum_steps = int(train_cfg['accum_steps'])
    num_workers = int(data_cfg['num_workers'])
    effective_batch = micro_batch * accum_steps
    paired_sampler = None
    paired_settings = target_settings.get('paired_training') if target_settings else None
    consistency_weight = train_cfg.get('view_consistency_weight')
    if consistency_weight is not None:
        consistency_weight = float(consistency_weight)
        if not math.isfinite(consistency_weight) or consistency_weight < 0:
            raise ValueError('View consistency weight must be finite and nonnegative')
        if (paired_settings or (target_settings and target_settings['match_train'])
                or any(aug_cfg.get(key) for key in ('domain_shift', 'jpeg_aug', 'color_jitter'))
                or train_cfg.get('mixup_alpha') or train_cfg.get('cutmix_alpha')):
            raise ValueError('Paired views define their own training augmentation')
        train_tf = PairedDegradation(data_cfg['input_size'], mean, std, aug_cfg.get('hflip', .5))
        print(f'[train] 成对视图训练，预测一致性权重={consistency_weight}')
    if paired_settings:
        paired_sampler = PairedTargetSampler(
            train_rows, profiles, repo_path(paired_settings['metadata']), seed,
            fraction=paired_settings['fraction'], max_scale=paired_settings['max_scale'],
        )
        native_aug = {key: value for key, value in aug_cfg.items() if key not in ('domain_shift', 'jpeg_aug')}
        native_tf = build_transform(data_cfg['input_size'], data_cfg['resize_size'], mean, std,
                                    train=True, aug=native_aug)
        train_loader = DataLoader(
            PairedTrainingImages(train_rows, profiles, native_tf, train_tf,
                                 other_blur_radius=float(paired_settings.get('other_blur_radius', 0.))),
            sampler=paired_sampler, batch_size=micro_batch, num_workers=num_workers,
            drop_last=True, pin_memory=device.type == 'cuda', persistent_workers=num_workers > 0,
            worker_init_fn=make_worker_init_fn(seed), generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_loader = build_loader(train_rows, train_tf, micro_batch, shuffle=True,
                                    num_workers=num_workers, seed=seed, drop_last=True,
                                    persistent=True)
    val_loader = build_loader(select_rows, val_tf, batch_size=64, shuffle=False,
                              num_workers=num_workers, seed=seed)
    print(f'[data] 微批量 {micro_batch} × 累积 {accum_steps} = 有效批量 {effective_batch}')

    jpeg_rows = ensure_jpeg_cache(
        select_rows, repo_path(data_cfg['jpeg_cache']),
        quality=int(data_cfg['jpeg_quality']), subsampling=int(data_cfg['jpeg_subsampling']),
        num_threads=num_workers)
    val_jpeg_loader = build_loader(jpeg_rows, val_tf, batch_size=64, shuffle=False,
                                   num_workers=num_workers, seed=seed)
    print(f'[data] 验证集 JPEG 重编码副本: {repo_path(data_cfg["jpeg_cache"])}')

    # ------------------------------------------------------------------ 模型与优化
    model = build_model(model_cfg)
    model.to(device)
    param_total = sum(p.numel() for p in model.parameters())
    param_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[model] {model_cfg["name"]}: 参数量 {param_total / 1e6:.3f}M '
          f'(可训练 {param_trainable / 1e6:.3f}M)，随机初始化，未加载任何预训练权重')

    criterion = build_criterion(train_cfg)
    print(f'[train] 损失 {type(criterion).__name__} | mixup '
          f'{train_cfg.get("mixup_alpha", 0.0)} | cutmix {train_cfg.get("cutmix_alpha", 0.0)}')
    optimizer = build_optimizer(model, float(train_cfg['lr']),
                                float(train_cfg['weight_decay']),
                                tuple(train_cfg['betas']), float(train_cfg['eps']))
    amp, amp_dtype = resolve_amp(bool(train_cfg['amp']), train_cfg['amp_dtype'], device)
    scaler = GradScaler(device.type, enabled=amp)
    ema_decay = float(train_cfg.get('ema_decay', 0.0) or 0.0)
    ema = ModelEma(model, ema_decay) if ema_decay > 0 else None
    eval_model = ema.module if ema is not None else model
    print(f'[train] AMP={amp} dtype={amp_dtype} seed={seed} EMA={ema_decay or "off"}')

    epochs = int(train_cfg['epochs'])
    updates_per_epoch = int(train_cfg['updates_per_epoch'])
    total_updates = epochs * updates_per_epoch
    selection = str(train_cfg.get('selection', 'orig_auroc'))
    selection_desc = {'orig_auroc': 'val AUROC (原图)',
                      'jpeg_auroc': 'val AUROC (统一 JPEG 重编码)',
                      'mean_auroc': '平均 AUROC (原图 + JPEG)',
                      'proxy_worst_acc': '最差跨域代理准确率',
                      'target_worst_acc': '目标风格代理两批次中较低的准确率'}.get(selection, selection)
    proxy_loaders = {}
    if selection == 'proxy_worst_acc':
        proxy_loaders = {
            case: DataLoader(ImageDataset(val_rows, val_tf, FixedDomainShift(case)),
                             batch_size=micro_batch, num_workers=num_workers,
                             pin_memory=device.type == 'cuda', persistent_workers=num_workers > 0)
            for case in PROXY_CASES
        }
    if selection == 'target_worst_acc':
        if not target_settings:
            raise ValueError('Target proxy selection requires target_alignment settings')
        match = FixedTargetMatch(val_rows, profiles, 'both', seed=221)
        proxy_loaders = {
            cohort: DataLoader(
                ImageDataset([row for row in select_rows if match.cohort(row['image_id']) == cohort], val_tf, match),
                batch_size=micro_batch, num_workers=num_workers, pin_memory=device.type == 'cuda',
                persistent_workers=num_workers > 0,
            )
            for cohort in ('width500', 'other')
        }
        if target_settings.get('select_legacy_strong'):
            proxy_loaders['legacy_strong'] = DataLoader(
                ImageDataset(select_rows, val_tf, legacy_strong),
                batch_size=micro_batch, num_workers=num_workers,
                pin_memory=device.type == 'cuda', persistent_workers=num_workers > 0,
            )
            selection_desc = '目标风格两批次与强退化代理中的最差准确率'
    proxy_every = int(target_settings.get('select_every', 1)) if target_settings else 1
    if proxy_every < 1:
        raise ValueError('select_every must be at least 1')
    print(f'[train] 选模指标 {selection_desc}')
    print(f'[train] 计划 {epochs} epoch × {updates_per_epoch} 次更新 = {total_updates} 次')

    run_info = {
        'run_name': cfg['output']['run_name'],
        'config': cfg,
        'environment': env,
        'params_total': param_total,
        'params_trainable': param_trainable,
        'micro_batch': micro_batch,
        'accum_steps': accum_steps,
        'effective_batch': effective_batch,
        'planned_updates': total_updates,
        'norm_stats': stats,
        'num_train': len(train_rows),
        'num_val': len(val_rows),
    }
    with open(run_dir / 'run_info.json', 'w', encoding='utf-8') as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
        f.write('\n')


    # ------------------------------------------------------------------ 训练
    best = {'score': float('-inf'), 'auroc': float('nan'), 'epoch': -1, 'metrics': None}
    history = []
    train_start = time.time()
    for epoch in range(1, epochs + 1):
        if paired_sampler is not None:
            paired_sampler.set_epoch(epoch)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        train_log = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, amp, amp_dtype,
            accum_steps=accum_steps, updates_per_epoch=updates_per_epoch,
            total_updates=total_updates, base_lr=float(train_cfg['lr']),
            start_lr=float(train_cfg['lr_start']), final_lr=float(train_cfg['lr_final']),
            warmup_updates=int(train_cfg['warmup_updates']),
            global_update=(epoch - 1) * updates_per_epoch,
            mixup_alpha=float(train_cfg.get('mixup_alpha', 0.0) or 0.0),
            cutmix_alpha=float(train_cfg.get('cutmix_alpha', 0.0) or 0.0),
            mix_prob=float(train_cfg.get('mix_prob', 1.0)),
            ema=ema, num_classes=int(model_cfg['num_classes']),
            consistency_weight=consistency_weight or 0.0,
        )
        val_out = evaluate(eval_model, val_loader, criterion, device, amp, amp_dtype)
        val_metrics = val_out['metrics']
        raw_metrics = None
        if ema is not None and not target_settings:
            raw_metrics = evaluate(model, val_loader, criterion, device, amp, amp_dtype)['metrics']
        jpeg_metrics = None
        if selection in ('jpeg_auroc', 'mean_auroc'):
            jpeg_metrics = evaluate(eval_model, val_jpeg_loader, criterion, device, amp,
                                    amp_dtype)['metrics']
        proxy_due = epoch % proxy_every == 0 or epoch == epochs
        proxy_metrics = {}
        if proxy_due:
            proxy_metrics = {case: evaluate(eval_model, loader, criterion, device, amp, amp_dtype)['metrics']
                             for case, loader in proxy_loaders.items()}
        if proxy_metrics:
            select_score = min(metrics['accuracy'] for metrics in proxy_metrics.values())
        elif selection == 'jpeg_auroc' and jpeg_metrics is not None:
            select_score = (jpeg_metrics['auroc'] if math.isfinite(jpeg_metrics['auroc'])
                            else jpeg_metrics['accuracy'])
        elif selection == 'mean_auroc' and jpeg_metrics is not None:
            vals = [m['auroc'] for m in (val_metrics, jpeg_metrics) if math.isfinite(m['auroc'])]
            select_score = sum(vals) / len(vals) if vals else val_metrics['accuracy']
        elif proxy_loaders:
            select_score = None
        else:
            select_score = (val_metrics['auroc'] if math.isfinite(val_metrics['auroc'])
                            else val_metrics['accuracy'])

        record = {
            'epoch': epoch,
            'train': train_log,
            'val': {k: v for k, v in val_metrics.items() if k != 'confusion_matrix'},
            'val_confusion_matrix': val_metrics['confusion_matrix'],
            'select_score': select_score,
            'elapsed_seconds': time.time() - train_start,
        }
        if raw_metrics is not None:
            record['val_raw'] = {k: v for k, v in raw_metrics.items()
                                 if k != 'confusion_matrix'}
        if jpeg_metrics is not None:
            record['val_jpeg'] = {k: v for k, v in jpeg_metrics.items()
                                  if k != 'confusion_matrix'}
        if proxy_metrics:
            record['val_proxy'] = proxy_metrics
            print('[proxy] ' + ' '.join(f'{case}={metrics["accuracy"]:.4f}'
                                       for case, metrics in proxy_metrics.items()), flush=True)
        append_jsonl(log_path, record)
        history.append(record)
        jpeg_str = (f" | jpeg acc {jpeg_metrics['accuracy']:.4f} auroc {jpeg_metrics['auroc']:.4f}"
                    if jpeg_metrics is not None else '')
        print(f"[epoch {epoch:02d}/{epochs}] "
              f"train loss {train_log['loss']:.4f} acc {train_log['accuracy']:.4f} | "
              f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
              f"auroc {val_metrics['auroc']:.4f} "
              f"(R0 {val_metrics['recall_real']:.4f} R1 {val_metrics['recall_fake']:.4f})"
              f"{jpeg_str} | "
              f"lr {train_log['lr']:.2e} | {train_log['seconds']:.1f}s | "
              f"peak {train_log['peak_memory_mb']:.0f}MB", flush=True)

        extra = {'param_count': param_total, 'effective_batch': effective_batch,
                 'weights': 'ema' if ema is not None else 'raw'}
        save_checkpoint(run_dir / 'last.pt', eval_model, optimizer, epoch, val_metrics, cfg,
                        extra=extra)
        if select_score is not None and select_score > best['score']:
            best = {'score': select_score, 'auroc': val_metrics['auroc'], 'epoch': epoch,
                    'metrics': val_metrics}
            save_checkpoint(run_dir / 'best.pt', eval_model, optimizer, epoch, val_metrics,
                            cfg, extra=extra)
            print(f'[ckpt] 新最佳 epoch {epoch} ({selection_desc}: {select_score:.4f}) '
                  f'-> {run_dir / "best.pt"}', flush=True)

    total_minutes = (time.time() - train_start) / 60
    print(f'[train] 训练完成，用时 {total_minutes:.2f} 分钟；最佳 epoch {best["epoch"]} '
          f'val AUROC {best["auroc"]:.4f}')

    # ------------------------------------------------------------------ 用最佳权重评测原图与 JPEG 版
    best_path = run_dir / 'best.pt'
    payload = load_checkpoint(best_path, eval_model, device)
    print(f'[eval] 载入最佳 checkpoint (epoch {payload["epoch"]}) 评测原图与 JPEG 重编码验证集')
    orig_out = evaluate(eval_model, val_loader, criterion, device, amp, amp_dtype)
    jpeg_out = evaluate(eval_model, val_jpeg_loader, criterion, device, amp, amp_dtype)
    orig_out['metrics']['accuracy_best_threshold'], orig_out['metrics']['best_threshold'] = \
        best_threshold_accuracy(orig_out['labels'], orig_out['probs'])

    for tag, rows, out in (('orig', select_rows, orig_out), ('jpeg', jpeg_rows, jpeg_out)):
        prob_path = run_dir / f'val_predictions_{tag}.csv'
        with open(prob_path, 'w', encoding='utf-8') as f:
            f.write('image_id,label,prob_fake\n')
            for row, prob in zip(rows, out['probs'][:, 1]):
                f.write(f"{row['image_id']},{row['label']},{prob:.6f}\n")

    peak_mb = max(r['train']['peak_memory_mb'] for r in history)
    epoch_seconds = [r['train']['seconds'] for r in history]
    summary = {
        'run_name': cfg['output']['run_name'],
        'config': cfg,
        'environment': env,
        'git_commit': env['git_commit'],
        'params_total': param_total,
        'params_trainable': param_trainable,
        'micro_batch': micro_batch,
        'accum_steps': accum_steps,
        'effective_batch': effective_batch,
        'epochs': epochs,
        'updates_per_epoch': updates_per_epoch,
        'total_updates': total_updates,
        'train_minutes': total_minutes,
        'epoch_train_seconds_mean': sum(epoch_seconds) / len(epoch_seconds),
        'epoch_train_seconds_min': min(epoch_seconds),
        'epoch_train_seconds_max': max(epoch_seconds),
        'peak_memory_mb': peak_mb,
        'best_epoch': best['epoch'],
        'best_val_auroc_orig': best['auroc'],
        'best_selection_metric': selection_desc,
        'val_orig': orig_out['metrics'],
        'val_jpeg_q75': jpeg_out['metrics'],
        **({'val_proxy': history[best['epoch'] - 1]['val_proxy']} if proxy_loaders else {}),
        'checkpoint_best': str(best_path),
        'checkpoint_last': str(run_dir / 'last.pt'),
        'log': str(log_path),
        'norm_stats': {'path': str(norm_path), **stats},
    }
    with open(run_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write('\n')

    print('\n[eval] 最佳 checkpoint 的验证结果')
    header = f"{'验证版本':<12}{'acc':>8}{'auroc':>8}{'R_real':>8}{'R_fake':>8}   混淆矩阵[[TN,FP],[FN,TP]]"
    print(header)
    for tag, metrics in (('orig', orig_out['metrics']), ('jpeg_q75', jpeg_out['metrics'])):
        print(f"{tag:<12}{metrics['accuracy']:>8.4f}{metrics['auroc']:>8.4f}"
              f"{metrics['recall_real']:>8.4f}{metrics['recall_fake']:>8.4f}   "
              f"{metrics['confusion_matrix']}")
    print(f"[eval] 参数量 {param_total / 1e6:.3f}M，微批量 {micro_batch}×{accum_steps}="
          f"{effective_batch}，峰值显存 {peak_mb:.0f}MB，"
          f"每轮训练 {summary['epoch_train_seconds_mean']:.1f}s")
    print(f'[done] 汇总写入 {run_dir / "results.json"}')


if __name__ == '__main__':
    main()

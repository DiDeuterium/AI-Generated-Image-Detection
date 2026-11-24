"""多质量 JPEG 代理评测：把验证集按若干质量重编码成 JPEG（对齐测试集的编码方式），
在每条质量曲线上报告准确率、平衡准确率、AUROC 与两类召回，并给出原图（原生格式）参考。

用法（仓库根目录）：
    python fqm/evaluate_proxy.py --config <cfg.json> --checkpoint <best.pt> \
        --qualities 60 75 90 [--tta]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssd.data import (build_loader, build_transform, ensure_jpeg_cache, load_norm_stats,
                      read_splits)
from ssd.engine import (best_threshold_accuracy, compute_metrics, load_checkpoint, predict,
                        resolve_amp)
from ssd.tta import TTADataset, TTATransform, evaluate_tta
from train import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description='多质量 JPEG 代理评测')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--qualities', type=int, nargs='+', default=[60, 75, 90])
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--tta-scales', type=int, nargs='+', default=None,
                        help='多尺度 TTA：例如 320 352，会对每个尺度做五裁剪+翻转再平均')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=5)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    data_cfg = cfg['data']
    aug = data_cfg.get('aug', {})
    stats = load_norm_stats(REPO_ROOT / data_cfg['norm_stats'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp, amp_dtype = resolve_amp(bool(cfg['train']['amp']), cfg['train']['amp_dtype'], device)

    _, val_rows = read_splits(REPO_ROOT / data_cfg['splits'], REPO_ROOT / data_cfg['image_dir'])
    labels = np.array([r['label'] for r in val_rows])

    model = build_model(cfg['model']).to(device)
    payload = load_checkpoint(ckpt_path, model, device)
    print(f'[proxy] {ckpt_path} (epoch {payload.get("epoch")}) tta={args.tta}', flush=True)

    if args.tta or args.tta_scales:
        scales = args.tta_scales or [data_cfg['input_size']]
        transforms = []
        for scale in scales:
            size = int(scale)
            resize = int(round(size * data_cfg['resize_size'] / data_cfg['input_size']))
            transforms.append(TTATransform(size, resize, stats['mean'], stats['std'],
                                           mode=aug.get('mode', 'protocol')))
        make_loader = lambda rows, tf: DataLoader(TTADataset(rows, tf),
                                                  batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)

        def infer(rows):
            acc = None
            for tf in transforms:
                probs = evaluate_tta(model, make_loader(rows, tf), device, amp,
                                     amp_dtype)['probs']
                acc = probs if acc is None else acc + probs
            return acc / len(transforms)
    else:
        transform = build_transform(data_cfg['input_size'], data_cfg['resize_size'],
                                    stats['mean'], stats['std'], train=False, aug=aug)
        make_loader = lambda rows: build_loader(rows, transform, 64, False,
                                                args.num_workers, 0)
        infer = lambda rows: predict(model, make_loader(rows), device, amp, amp_dtype)

    results = {}
    variants = [('orig', val_rows)]
    for quality in args.qualities:
        cache = REPO_ROOT / data_cfg['jpeg_cache'].replace('val_jpeg_q75', f'val_jpeg_q{quality}')
        variants.append((f'q{quality}', ensure_jpeg_cache(val_rows, cache, quality=quality,
                                                          subsampling=2,
                                                          num_threads=args.num_workers)))
    for tag, rows in variants:
        probs = infer(rows)
        metrics = compute_metrics(labels, probs)
        balanced = 0.5 * (metrics['recall_real'] + metrics['recall_fake'])
        acc_thr, thr = best_threshold_accuracy(labels, probs)
        metrics.update({'balanced_accuracy': balanced,
                        'accuracy_best_threshold': acc_thr, 'best_threshold': thr})
        results[tag] = metrics
        print(f"[proxy] {tag:<5} acc {metrics['accuracy']:.4f} bal {balanced:.4f} "
              f"auroc {metrics['auroc']:.4f} R0 {metrics['recall_real']:.4f} "
              f"R1 {metrics['recall_fake']:.4f} acc@thr {acc_thr:.4f}", flush=True)

    q = [results[f'q{quality}'] for quality in args.qualities]
    summary = {
        'checkpoint': str(ckpt_path),
        'tta': args.tta,
        'qualities': args.qualities,
        'mean_quality_accuracy': float(np.mean([m['accuracy'] for m in q])),
        'mean_quality_balanced_accuracy': float(np.mean([m['balanced_accuracy'] for m in q])),
        'mean_quality_auroc': float(np.mean([m['auroc'] for m in q])),
        'worst_quality_accuracy': float(np.min([m['accuracy'] for m in q])),
        'metrics': results,
    }
    out_path = (args.output if args.output is not None
                else ckpt_path.parent / 'proxy_results.json')
    out_path = out_path if out_path.is_absolute() else REPO_ROOT / out_path
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f"[proxy] 多质量均值：acc {summary['mean_quality_accuracy']:.4f} "
          f"bal {summary['mean_quality_balanced_accuracy']:.4f} "
          f"auroc {summary['mean_quality_auroc']:.4f}；写入 {out_path}")


if __name__ == '__main__':
    main()

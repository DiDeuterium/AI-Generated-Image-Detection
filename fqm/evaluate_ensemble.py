"""多种子 checkpoint 概率集成评测（可选 TTA），输出原图 / JPEG 版指标。

用法（仓库根目录）：
    python fqm/evaluate_ensemble.py --config fqm/configs/efficientvim_cap.json \
        --checkpoints fqm/runs/final/x_seed42/best.pt fqm/runs/final/x_seed17/best.pt --tta
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssd.data import build_loader, build_transform, ensure_jpeg_cache, load_norm_stats, read_splits
from ssd.engine import (best_threshold_accuracy, compute_metrics, load_checkpoint, predict,
                        resolve_amp)
from ssd.tta import TTADataset, TTATransform, evaluate_tta
from train import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description='多种子集成评测')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=5)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    data_cfg = cfg['data']
    stats = load_norm_stats(REPO_ROOT / data_cfg['norm_stats'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp, amp_dtype = resolve_amp(bool(cfg['train']['amp']), cfg['train']['amp_dtype'], device)

    _, val_rows = read_splits(REPO_ROOT / data_cfg['splits'], REPO_ROOT / data_cfg['image_dir'])
    jpeg_rows = ensure_jpeg_cache(val_rows, REPO_ROOT / data_cfg['jpeg_cache'],
                                  quality=int(data_cfg['jpeg_quality']),
                                  subsampling=int(data_cfg['jpeg_subsampling']))
    labels = np.array([r['label'] for r in val_rows])

    if args.tta:
        transform = TTATransform(data_cfg['input_size'], data_cfg['resize_size'],
                                 stats['mean'], stats['std'],
                                 mode=data_cfg.get('aug', {}).get('mode', 'protocol'))
        make_loader = lambda rows: DataLoader(TTADataset(rows, transform),
                                              batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
        infer = lambda model, loader: evaluate_tta(model, loader, device, amp, amp_dtype)['probs']
    else:
        transform = build_transform(data_cfg['input_size'], data_cfg['resize_size'],
                                    stats['mean'], stats['std'], train=False,
                                    aug=data_cfg.get('aug', {}))
        make_loader = lambda rows: build_loader(rows, transform, 64, False, args.num_workers, 0)
        infer = lambda model, loader: predict(model, loader, device, amp, amp_dtype)

    results = {}
    for tag, rows in (('orig', val_rows), ('jpeg', jpeg_rows)):
        loader = make_loader(rows)
        probs_sum = None
        for ckpt in args.checkpoints:
            ckpt_path = ckpt if ckpt.is_absolute() else REPO_ROOT / ckpt
            model = build_model(cfg['model']).to(device)
            payload = load_checkpoint(ckpt_path, model, device)
            probs = infer(model, loader)
            print(f'[ens] {tag} {ckpt_path.parent.name} epoch {payload.get("epoch")} '
                  f'acc {compute_metrics(labels, probs)["accuracy"]:.4f}', flush=True)
            probs_sum = probs if probs_sum is None else probs_sum + probs
            del model
            torch.cuda.empty_cache()
        probs = probs_sum / len(args.checkpoints)
        metrics = compute_metrics(labels, probs)
        metrics['accuracy_best_threshold'], metrics['best_threshold'] = \
            best_threshold_accuracy(labels, probs)
        results[tag] = metrics
        print(f"[ens] == {tag} 集成: acc {metrics['accuracy']:.4f} auroc {metrics['auroc']:.4f} "
              f"R_real {metrics['recall_real']:.4f} R_fake {metrics['recall_fake']:.4f} "
              f"acc@thr {metrics['accuracy_best_threshold']:.4f} cm {metrics['confusion_matrix']}")

    out_path = args.output or (args.checkpoints[0].parent.parent / 'ensemble_results.json')
    out_path = out_path if out_path.is_absolute() else REPO_ROOT / out_path
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'tta': args.tta, 'checkpoints': [str(c) for c in args.checkpoints],
                   'metrics': results}, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'[ens] 写入 {out_path}')


if __name__ == '__main__':
    main()

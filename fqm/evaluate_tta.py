"""对已训练 checkpoint 做多视图 TTA 评测（五裁剪 + 翻转），并在原图与 JPEG 版上报告指标。

用法（仓库根目录）：
    python fqm/evaluate_tta.py --config fqm/tune/configs/w2_native_ema.json \
        --checkpoint fqm/runs/tune/w2_native_ema/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssd.data import ensure_jpeg_cache, load_norm_stats, read_splits
from ssd.engine import (best_threshold_accuracy, compute_metrics, load_checkpoint,
                        resolve_amp)
from ssd.tta import TTADataset, TTATransform, evaluate_tta
from train import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description='多视图 TTA 评测')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--no-flip', action='store_true', help='只用五裁剪，不翻转')
    parser.add_argument('--limit', type=int, default=None,
                        help='只用前 N 张验证图（调试用）')
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    ckpt_path = (args.checkpoint if args.checkpoint.is_absolute()
                 else REPO_ROOT / args.checkpoint)
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    data_cfg = cfg['data']
    aug = data_cfg.get('aug', {})
    stats = load_norm_stats(REPO_ROOT / data_cfg['norm_stats'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp, amp_dtype = resolve_amp(bool(cfg['train']['amp']), cfg['train']['amp_dtype'], device)
    model = build_model(cfg['model']).to(device)
    payload = load_checkpoint(ckpt_path, model, device)
    print(f'[tta] checkpoint {ckpt_path} (epoch {payload.get("epoch")})')

    train_rows, val_rows = read_splits(REPO_ROOT / data_cfg['splits'],
                                       REPO_ROOT / data_cfg['image_dir'])
    if args.limit:
        val_rows = val_rows[:args.limit]
    jpeg_rows = ensure_jpeg_cache(val_rows, REPO_ROOT / data_cfg['jpeg_cache'],
                                  quality=int(data_cfg['jpeg_quality']),
                                  subsampling=int(data_cfg['jpeg_subsampling']))
    transform = TTATransform(data_cfg['input_size'], data_cfg['resize_size'],
                             stats['mean'], stats['std'],
                             mode=aug.get('mode', 'protocol'), flip=not args.no_flip)

    results = {}
    for tag, rows in (('orig', val_rows), ('jpeg', jpeg_rows)):
        loader = DataLoader(TTADataset(rows, transform), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
        out = evaluate_tta(model, loader, device, amp, amp_dtype)
        metrics = compute_metrics(out['labels'], out['probs'])
        acc_thr, thr = best_threshold_accuracy(out['labels'], out['probs'])
        metrics['accuracy_best_threshold'] = acc_thr
        metrics['best_threshold'] = thr
        results[tag] = metrics
        print(f"[tta] {tag:<5} acc {metrics['accuracy']:.4f} auroc {metrics['auroc']:.4f} "
              f"R_real {metrics['recall_real']:.4f} R_fake {metrics['recall_fake']:.4f} "
              f"acc@thr {acc_thr:.4f} cm {metrics['confusion_matrix']}")

    out_path = ckpt_path.parent / 'tta_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'[tta] 写入 {out_path}')


if __name__ == '__main__':
    main()

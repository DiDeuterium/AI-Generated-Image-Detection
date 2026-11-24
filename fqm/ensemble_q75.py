"""集成 + 多尺度 TTA 的代理评测与提交导出（面向 q75 验证集与测试集）。

用法（仓库根目录）：
    python fqm/ensemble_q75.py --config <cfg> --checkpoints a.pt b.pt \
        --tta-scales 320 352 --qualities 60 75 90 [--submit out.csv]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssd.data import ensure_jpeg_cache, load_norm_stats, read_splits, read_test_ids
from ssd.engine import best_threshold_accuracy, compute_metrics, load_checkpoint, resolve_amp
from ssd.tta import TTADataset, TTATransform, evaluate_tta
from train import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description='集成 + 多尺度 TTA 的 q75 代理评测/提交')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--tta-scales', type=int, nargs='+', default=[320])
    parser.add_argument('--qualities', type=int, nargs='+', default=[75])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=5)
    parser.add_argument('--submit', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    data_cfg = cfg['data']
    aug = data_cfg.get('aug', {})
    stats = load_norm_stats(REPO_ROOT / data_cfg['norm_stats'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp, amp_dtype = resolve_amp(bool(cfg['train']['amp']), cfg['train']['amp_dtype'], device)

    transforms = []
    for scale in args.tta_scales:
        resize = int(round(scale * data_cfg['resize_size'] / data_cfg['input_size']))
        transforms.append(TTATransform(scale, resize, stats['mean'], stats['std'],
                                       mode=aug.get('mode', 'protocol')))

    def infer_all(rows):
        total, count = None, 0
        for tf in transforms:
            loader = DataLoader(TTADataset(rows, tf), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers, pin_memory=True)
            for ckpt in args.checkpoints:
                ckpt_path = ckpt if ckpt.is_absolute() else REPO_ROOT / ckpt
                model = build_model(cfg['model']).to(device)
                payload = load_checkpoint(ckpt_path, model, device)
                probs = evaluate_tta(model, loader, device, amp, amp_dtype)['probs']
                print(f'[ens] scale {tf.size} | {ckpt_path.parent.name} '
                      f'(ep {payload.get("epoch")}) fake比例 {(probs[:, 1] > 0.5).mean():.3f}',
                      flush=True)
                total = probs if total is None else total + probs
                count += 1
                del model
                torch.cuda.empty_cache()
        return total / count

    results = {}
    _, val_rows = read_splits(REPO_ROOT / data_cfg['splits'], REPO_ROOT / data_cfg['image_dir'])
    labels = np.array([r['label'] for r in val_rows])
    for quality in args.qualities:
        cache = REPO_ROOT / f'fqm/cache/val_jpeg_q{quality}'
        rows = ensure_jpeg_cache(val_rows, cache, quality=quality, subsampling=2,
                                 num_threads=args.num_workers)
        probs = infer_all(rows)
        metrics = compute_metrics(labels, probs)
        metrics['balanced_accuracy'] = 0.5 * (metrics['recall_real'] + metrics['recall_fake'])
        metrics['accuracy_best_threshold'], metrics['best_threshold'] = \
            best_threshold_accuracy(labels, probs)
        results[f'q{quality}'] = metrics
        print(f"[ens] == q{quality}: acc {metrics['accuracy']:.4f} "
              f"bal {metrics['balanced_accuracy']:.4f} auroc {metrics['auroc']:.4f} "
              f"R0 {metrics['recall_real']:.4f} R1 {metrics['recall_fake']:.4f} "
              f"acc@thr {metrics['accuracy_best_threshold']:.4f}", flush=True)


    out = {'checkpoints': [str(c) for c in args.checkpoints],
           'tta_scales': args.tta_scales, 'metrics': results}
    out_path = (args.output if args.output is not None
                else REPO_ROOT / 'fqm/runs/final/ensemble_q75_results.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'[ens] 写入 {out_path}')

    if args.submit is not None:
        template = (REPO_ROOT / 'ai-generated-image-detection-benchmark'
                    / 'sample_submission.csv')
        test_rows = read_test_ids(template,
                                  REPO_ROOT / data_cfg['image_dir'] / 'dataset' / 'test')
        probs = infer_all(test_rows)
        labels_pred = probs.argmax(axis=1)
        sub_path = args.submit if args.submit.is_absolute() else REPO_ROOT / args.submit
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sub_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['image_id', 'label'])
            for row, label in zip(test_rows, labels_pred):
                writer.writerow([row['image_id'], int(label)])
        print(f'[ens] 提交写入 {sub_path}: {len(test_rows)} 行，'
              f'判为 AI 生成 {int(labels_pred.sum())} 张')
        written = list(csv.DictReader(open(sub_path)))
        tpl = list(csv.DictReader(open(template)))
        assert [r['image_id'] for r in written] == [r['image_id'] for r in tpl]
        assert len(written) == len(tpl) == 1000
        assert len({r['image_id'] for r in written}) == 1000
        assert all(r['label'] in ('0', '1') for r in written)
        print('[ens] 提交校验通过：1000 行、ID 顺序与模板一致、标签 0/1')


if __name__ == '__main__':
    main()

"""多 checkpoint（可叠加 TTA）对测试集推理，按比赛模板导出提交文件。

用法（仓库根目录）：
    python fqm/predict_ensemble.py --config fqm/configs/efficientvim_best.json \
        --checkpoints fqm/runs/final/final_w5_cap60_seed42/best.pt ... --tta \
        --output fqm/runs/final/submission_ensemble.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssd.data import (build_loader, build_transform, load_norm_stats, read_test_ids)
from ssd.engine import load_checkpoint, predict, resolve_amp
from ssd.tta import TTADataset, TTATransform, evaluate_tta
from train import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description='集成推理测试集')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--no-flip', action='store_true', help='多裁剪时使用五个原方向视图')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    data_cfg = cfg['data']
    stats = load_norm_stats(REPO_ROOT / data_cfg['norm_stats'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp, amp_dtype = resolve_amp(bool(cfg['train']['amp']), cfg['train']['amp_dtype'], device)

    template = REPO_ROOT / 'ai-generated-image-detection-benchmark' / 'sample_submission.csv'
    test_dir = REPO_ROOT / data_cfg['image_dir'] / 'dataset' / 'test'
    rows = read_test_ids(template, test_dir)
    if args.tta:
        transform = TTATransform(data_cfg['input_size'], data_cfg['resize_size'],
                                 stats['mean'], stats['std'],
                                 mode=data_cfg.get('aug', {}).get('mode', 'protocol'), flip=not args.no_flip)
        loader = DataLoader(TTADataset(rows, transform), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
        infer = lambda model: evaluate_tta(model, loader, device, amp, amp_dtype)['probs']
    else:
        transform = build_transform(data_cfg['input_size'], data_cfg['resize_size'],
                                    stats['mean'], stats['std'], train=False,
                                    aug=data_cfg.get('aug', {}))
        loader = build_loader(rows, transform, 64, False, args.num_workers, 0)
        infer = lambda model: predict(model, loader, device, amp, amp_dtype)

    probs_sum = None
    for ckpt in args.checkpoints:
        ckpt_path = ckpt if ckpt.is_absolute() else REPO_ROOT / ckpt
        model = build_model(cfg['model']).to(device)
        payload = load_checkpoint(ckpt_path, model, device)
        probs = infer(model)
        print(f'[ens-pred] {ckpt_path.parent.name} epoch {payload.get("epoch")} '
              f'fake比例 {(probs[:, 1] > 0.5).mean():.3f}', flush=True)
        probs_sum = probs if probs_sum is None else probs_sum + probs
        del model
        torch.cuda.empty_cache()

    probs = probs_sum / len(args.checkpoints)
    labels = probs.argmax(axis=1)
    out_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'label'])
        for row, label in zip(rows, labels):
            writer.writerow([row['image_id'], int(label)])
    print(f'[ens-pred] 写入 {out_path}: {len(rows)} 行，预测 AI 生成 {int(labels.sum())} 张')

    written = list(csv.DictReader(open(out_path)))
    tpl = list(csv.DictReader(open(template)))
    assert [r['image_id'] for r in written] == [r['image_id'] for r in tpl]
    assert len(written) == len(tpl) == 1000 and len({r['image_id'] for r in written}) == 1000
    assert all(r['label'] in ('0', '1') for r in written)
    print('[ens-pred] 校验通过：1000 行、ID 顺序与模板一致、标签 0/1')


if __name__ == '__main__':
    main()

"""用训练好的最佳 checkpoint 对测试集推理并按比赛模板导出提交文件。

测试图只用于最终推理，不参与训练、选模或调参。
用法（仓库根目录）：
    python fqm/predict.py --config fqm/configs/efficientvim_m1.json \
        --checkpoint fqm/runs/efficientvim_m1_seed42/best.pt
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssd.data import build_loader, build_transform, load_norm_stats, read_test_ids
from ssd.engine import load_checkpoint, predict, resolve_amp
from train import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='EfficientViM 测试集推理')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None,
                        help='提交文件路径，默认写在 checkpoint 同目录 submission.csv')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=None,
                        help='只推理前 N 张（调试用，正式提交不要使用）')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    ckpt_path = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    with open(cfg_path, encoding='utf-8') as f:
        cfg = json.load(f)
    data_cfg = cfg['data']

    stats = load_norm_stats(REPO_ROOT / data_cfg['norm_stats'])
    if stats is None:
        raise FileNotFoundError('缺少归一化统计，请先运行 fqm/train.py 生成')
    transform = build_transform(data_cfg['input_size'], data_cfg['resize_size'],
                                stats['mean'], stats['std'], train=False,
                                aug=data_cfg.get('aug', {}))

    template = REPO_ROOT / 'ai-generated-image-detection-benchmark' / 'sample_submission.csv'
    rows = read_test_ids(template, REPO_ROOT / data_cfg['image_dir'] / 'dataset' / 'test')
    if args.limit:
        rows = rows[:args.limit]
    loader = build_loader(rows, transform, args.batch_size, shuffle=False,
                          num_workers=data_cfg['num_workers'], seed=0)
    print(f'[predict] 测试图 {len(rows)} 张，checkpoint {ckpt_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg['model']).to(device)
    payload = load_checkpoint(ckpt_path, model, device)
    print(f'[predict] 载入 epoch {payload.get("epoch")} 的权重')
    amp, amp_dtype = resolve_amp(bool(cfg['train']['amp']), cfg['train']['amp_dtype'], device)

    probs = predict(model, loader, device, amp, amp_dtype)
    labels = probs.argmax(axis=1)
    out_path = args.output or (ckpt_path.parent / 'submission.csv')
    out_path = out_path if out_path.is_absolute() else REPO_ROOT / out_path
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'label'])
        for row, label in zip(rows, labels):
            writer.writerow([row['image_id'], int(label)])
    print(f'[predict] 写入 {out_path}: {len(rows)} 行，预测为 AI 生成 {int(labels.sum())} 张')


if __name__ == '__main__':
    main()

"""阈值校准与诚实评估：把验证集按 image_id 哈希一分为二，A 半找最优阈值、B 半报告指标。

避免"在同一份验证集上既调阈值又报准确率"带来的乐观偏差。

用法（仓库根目录）：
    python fqm/calibrate_threshold.py \
        --pred-csv fqm/runs/tune/w6_cap60_jpegaug/val_predictions_jpeg.csv \
        --out fqm/runs/tune/w6_cap60_jpegaug/calibrated.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_preds(path: Path):
    import csv
    labels, probs, ids = [], [], []
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            ids.append(row['image_id'])
            labels.append(int(row['label']))
            probs.append(float(row['prob_fake']))
    return ids, np.array(labels), np.array(probs)


def best_threshold(labels: np.ndarray, probs: np.ndarray):
    """在给定子集上扫描阈值，返回使准确率最高的阈值。"""
    order = np.unique(probs)
    best_t, best_acc = 0.5, -1.0
    for t in order:
        acc = float(((probs >= t).astype(int) == labels).mean())
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t, best_acc


def accuracy_at(labels: np.ndarray, probs: np.ndarray, threshold: float):
    pred = (probs >= threshold).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for y, p in zip(labels, pred):
        cm[y, p] += 1
    recall0 = cm[0, 0] / max(1, cm[0].sum())
    recall1 = cm[1, 1] / max(1, cm[1].sum())
    return float((pred == labels).mean()), float(recall0), float(recall1), cm.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description='阈值校准（半分验证）')
    parser.add_argument('--pred-csv', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    csv_path = args.pred_csv if args.pred_csv.is_absolute() else REPO_ROOT / args.pred_csv
    ids, labels, probs = load_preds(csv_path)

    fold = np.array([int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) % 2 for i in ids])
    a, b = fold == 0, fold == 1
    tau, acc_a = best_threshold(labels[a], probs[a])
    acc_b, r0, r1, cm = accuracy_at(labels[b], probs[b], tau)
    acc_all_at_05, r0_05, r1_05, cm_05 = accuracy_at(labels, probs, 0.5)
    acc_b_at_05, _, _, _ = accuracy_at(labels[b], probs[b], 0.5)

    report = {
        'pred_csv': str(csv_path),
        'n': int(len(labels)),
        'calibration_half': {
            'n': int(a.sum()), 'threshold': tau, 'accuracy_in_half': acc_a},
        'heldout_half_at_calibrated_threshold': {
            'n': int(b.sum()), 'accuracy': acc_b, 'recall_real': r0, 'recall_fake': r1,
            'confusion_matrix': cm},
        'heldout_half_at_0.5': {'accuracy': acc_b_at_05},
        'all_at_0.5': {'accuracy': acc_all_at_05, 'recall_real': r0_05,
                       'recall_fake': r1_05, 'confusion_matrix': cm_05},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out is not None:
        out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write('\n')
        print(f'[calib] 写入 {out}')


if __name__ == '__main__':
    main()

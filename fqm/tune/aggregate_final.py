"""汇总最终复跑：每个种子的指标 + 跨种子的均值与样本标准差（含 TTA 结果）。

用法（仓库根目录）：
    python fqm/tune/aggregate_final.py --root fqm/runs/final
"""

import argparse
import json
import statistics as stats
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return 'n/a'
    if len(values) == 1:
        return f'{values[0]:.4f}'
    return f'{stats.mean(values):.4f} ± {stats.stdev(values):.4f}'


def main() -> None:
    parser = argparse.ArgumentParser(description='汇总最终复跑结果')
    parser.add_argument('--root', type=str, default='fqm/runs/final')
    args = parser.parse_args()

    rows = []
    for path in sorted((REPO / args.root).glob('*/results.json')):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        tta = None
        tta_path = path.parent / 'tta_results.json'
        if tta_path.exists():
            with open(tta_path, encoding='utf-8') as f:
                tta = json.load(f)
        rows.append({
            'run': data['run_name'],
            'seed': data['config']['train']['seed'],
            'acc': data['val_orig']['accuracy'],
            'auroc': data['val_orig']['auroc'],
            'acc_thr': data['val_orig'].get('accuracy_best_threshold'),
            'r_real': data['val_orig']['recall_real'],
            'r_fake': data['val_orig']['recall_fake'],
            'jpeg_acc': data['val_jpeg_q75']['accuracy'],
            'jpeg_auroc': data['val_jpeg_q75']['auroc'],
            'tta_acc': (tta or {}).get('orig', {}).get('accuracy'),
            'tta_auroc': (tta or {}).get('orig', {}).get('auroc'),
            'tta_jpeg_acc': (tta or {}).get('jpeg', {}).get('accuracy'),
            'tta_jpeg_auroc': (tta or {}).get('jpeg', {}).get('auroc'),
            'minutes': data.get('train_minutes'),
            'params': data.get('params_total'),
            'peak': data.get('peak_memory_mb'),
            'best_epoch': data.get('best_epoch'),
        })
    if not rows:
        print('没有找到最终结果')
        return

    header = (f"{'run':<40}{'seed':>5}{'acc':>8}{'auroc':>8}{'R_real':>8}{'R_fake':>8}"
              f"{'jpeg_acc':>9}{'jpeg_auc':>9}{'tta_acc':>9}{'tta_auc':>9}")
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['run']:<40}{r['seed']:>5}{r['acc']:>8.4f}{r['auroc']:>8.4f}"
              f"{r['r_real']:>8.4f}{r['r_fake']:>8.4f}{r['jpeg_acc']:>9.4f}"
              f"{r['jpeg_auroc']:>9.4f}"
              f"{(r['tta_acc'] if r['tta_acc'] is not None else float('nan')):>9.4f}"
              f"{(r['tta_auroc'] if r['tta_auroc'] is not None else float('nan')):>9.4f}")

    print()
    print(f"种子数 {len(rows)}；均值 ± 样本标准差")
    for key, name in (('acc', '原图 acc'), ('auroc', '原图 AUROC'),
                      ('r_real', '原图 R_real'), ('r_fake', '原图 R_fake'),
                      ('jpeg_acc', 'JPEG acc'), ('jpeg_auroc', 'JPEG AUROC'),
                      ('tta_acc', 'TTA acc'), ('tta_auroc', 'TTA AUROC'),
                      ('tta_jpeg_acc', 'TTA JPEG acc'), ('tta_jpeg_auroc', 'TTA JPEG AUROC')):
        print(f'  {name:<12} {mean_std([r[key] for r in rows])}')
    print(f"  {'参数量':<12} {rows[0]['params'] / 1e6:.3f}M")
    print(f"  {'每轮峰值显存':<12} {max(r['peak'] for r in rows):.0f} MB")
    print(f"  {'训练分钟':<12} {mean_std([r['minutes'] for r in rows])}")


if __name__ == '__main__':
    main()

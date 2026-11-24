"""按 q75 代理指标挑选最优配置与 checkpoint，输出集成所需的清单。

规则：把 run 名末尾的 `_seed<数字>` 去掉作为"变体"，按变体的 q75 平均准确率排序；
取最优变体的全部种子（最多 4 个），不足则用次优变体补齐。

用法：python fqm/tune/pick_q75.py --out fqm/logs/ensemble_list.txt
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def collect():
    rows = []
    for path in list((REPO / 'fqm/runs/tune').glob('*/results.json')) + \
            list((REPO / 'fqm/runs/final').glob('*/results.json')):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        jpeg = data.get('val_jpeg_q75') or {}
        if not jpeg.get('accuracy'):
            continue
        ckpt = path.parent / 'best.pt'
        if not ckpt.exists():
            continue
        run = data['run_name']
        variant = re.sub(r'_seed\d+$', '', run)
        rows.append({'run': run, 'variant': variant, 'q75_acc': jpeg['accuracy'],
                     'q75_auc': jpeg.get('auroc', 0.0), 'ckpt': str(ckpt),
                     'config': str(REPO / 'fqm/tune/configs' / f'{run}.json')})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='按 q75 指标挑最优变体与 checkpoint')
    parser.add_argument('--out', type=Path, default=Path('fqm/logs/ensemble_list.txt'))
    parser.add_argument('--max-models', type=int, default=4)
    args = parser.parse_args()

    rows = collect()
    if not rows:
        raise SystemExit('没有找到带 q75 指标的结果')

    groups = defaultdict(list)
    for row in rows:
        groups[row['variant']].append(row)
    ranked = sorted(groups.items(),
                    key=lambda kv: (sum(r['q75_acc'] for r in kv[1]) / len(kv[1])),
                    reverse=True)

    print('[pick] 变体排名（按 q75 平均准确率）')
    for name, members in ranked:
        mean_acc = sum(m['q75_acc'] for m in members) / len(members)
        print(f'  {name:<36} n={len(members)} mean_q75={mean_acc:.4f} '
              f'best={max(m["q75_acc"] for m in members):.4f}')

    picked, best_variant = [], None
    for name, members in ranked:
        members = sorted(members, key=lambda m: m['q75_acc'], reverse=True)
        for m in members:
            if len(picked) >= args.max_models:
                break
            picked.append(m)
        if picked and best_variant is None:
            best_variant = name
        if len(picked) >= args.max_models:
            break

    print(f'[pick] 选中变体 {best_variant}，共 {len(picked)} 个 checkpoint：')
    for m in picked:
        print(f"  {m['run']:<40} q75_acc {m['q75_acc']:.4f}")
    out = args.out if args.out.is_absolute() else REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(picked[0]['config'] + '\n')
        for m in picked:
            f.write(m['ckpt'] + '\n')
    print(f'[pick] 写入 {out}')


if __name__ == '__main__':
    main()

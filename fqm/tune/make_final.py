"""从调参结果中选出最优配置，生成最终复跑配置（多随机种子）与清单。

选择规则：先按原图验证 AUROC、再按准确率排序（与统一协议的选模指标一致）。
用法（仓库根目录）：
    python fqm/tune/make_final.py                # 默认种子 42,17,43,97
    python fqm/tune/make_final.py --seeds 42,17
"""

import argparse
import copy
import json
from pathlib import Path

TUNE = Path(__file__).resolve().parent
REPO = TUNE.parents[1]


def load_runs(root: Path):
    rows = []
    for path in sorted(root.glob('*/results.json')):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        orig = data['val_orig']
        rows.append({
            'run': data['run_name'],
            'acc': float(orig['accuracy']),
            'auroc': float(orig['auroc']),
            'config': data['config'],
            'params': data.get('params_total', 0),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='生成最终复跑配置')
    parser.add_argument('--root', type=str, default='fqm/runs/tune')
    parser.add_argument('--out-root', type=str, default='fqm/runs/final')
    parser.add_argument('--seeds', type=str, default='42,17,43,97')
    args = parser.parse_args()

    rows = load_runs(REPO / args.root)
    if not rows:
        raise SystemExit('没有找到 results.json')
    rows.sort(key=lambda r: (r['auroc'], r['acc']), reverse=True)
    best = rows[0]
    print(f"[final] 选中 {best['run']}: acc {best['acc']:.4f} auroc {best['auroc']:.4f} "
          f"({best['params'] / 1e6:.3f}M)")

    seeds = [int(s) for s in args.seeds.split(',')]
    base = copy.deepcopy(best['config'])
    base['output']['root'] = args.out_root
    manifest = []
    for seed in seeds:
        cfg = copy.deepcopy(base)
        cfg['train']['seed'] = seed
        cfg['output']['run_name'] = f"final_{best['run']}_seed{seed}"
        path = TUNE / 'configs' / f"{cfg['output']['run_name']}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write('\n')
        manifest.append(str(path.relative_to(REPO)))
        print(' ', manifest[-1])

    manifest_path = TUNE / 'manifest_final.txt'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(manifest) + '\n')

    canonical = copy.deepcopy(base)
    canonical['train']['seed'] = seeds[0]
    canonical['output'] = {'root': 'fqm/runs', 'run_name': 'efficientvim_best_seed42'}
    with open(REPO / 'fqm/configs/efficientvim_best.json', 'w', encoding='utf-8') as f:
        json.dump(canonical, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(f'[final] 清单 {manifest_path}；最佳配置副本 fqm/configs/efficientvim_best.json')


if __name__ == '__main__':
    main()

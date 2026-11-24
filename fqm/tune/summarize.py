"""汇总调参结果：读取各 run 的 results.json，按验证准确率排序打印表格。

用法（仓库根目录）：
    python fqm/tune/summarize.py                 # 汇总 fqm/runs/tune 下全部 run
    python fqm/tune/summarize.py --prefix w1_    # 只看第一轮
    python fqm/tune/summarize.py --sort auroc
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_runs(root: Path, prefix: str):
    runs = []
    for path in sorted(root.glob(f'{prefix}*/results.json')):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        cfg = data['config']
        train, model = cfg['train'], cfg['model']
        aug = cfg['data'].get('aug', {})
        orig, jpeg = data.get('val_orig', {}), data.get('val_jpeg_q75', {})
        runs.append({
            'run': data['run_name'],
            'acc': orig.get('accuracy', float('nan')),
            'auroc': orig.get('auroc', float('nan')),
            'acc_thr': orig.get('accuracy_best_threshold', float('nan')),
            'r_real': orig.get('recall_real', float('nan')),
            'r_fake': orig.get('recall_fake', float('nan')),
            'jpeg_acc': jpeg.get('accuracy', float('nan')),
            'jpeg_auroc': jpeg.get('auroc', float('nan')),
            'best_epoch': data.get('best_epoch'),
            'epochs': data.get('epochs'),
            'params': data.get('params_total', 0) / 1e6,
            'lr': train.get('lr'),
            'wd': train.get('weight_decay'),
            'ls': train.get('label_smoothing'),
            'loss': train.get('loss', 'ce'),
            'mix': train.get('mixup_alpha', 0.0),
            'ema': train.get('ema_decay', 0.0),
            'aug': aug.get('mode', 'protocol'),
            'input': cfg['data']['input_size'],
            'minutes': data.get('train_minutes', 0.0),
            'peak': data.get('peak_memory_mb', 0.0),
        })
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description='汇总调参结果')
    parser.add_argument('--root', type=str, default='fqm/runs/tune')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--sort', type=str, default='acc',
                        choices=['acc', 'auroc', 'acc_thr', 'jpeg_acc'])
    args = parser.parse_args()
    runs = load_runs(REPO_ROOT / args.root, args.prefix)
    if not runs:
        print('没有找到结果')
        return
    runs.sort(key=lambda r: r[args.sort], reverse=True)
    header = (f"{'run':<26}{'acc':>7}{'auroc':>7}{'acc@thr':>8}{'R_real':>8}{'R_fake':>7}"
              f"{'jpeg_acc':>9}{'jpeg_auc':>9}{'ep':>4}{'M':>6}{'min':>6}")
    print(header)
    print('-' * len(header))
    for r in runs:
        print(f"{r['run']:<26}{r['acc']:>7.4f}{r['auroc']:>7.4f}{r['acc_thr']:>8.4f}"
              f"{r['r_real']:>8.4f}{r['r_fake']:>7.4f}{r['jpeg_acc']:>9.4f}"
              f"{r['jpeg_auroc']:>9.4f}{r['best_epoch']:>4}{r['params']:>6.2f}"
              f"{r['minutes']:>6.1f}")
    print()
    print(f"{'run':<26}{'aug':>10}{'input':>7}{'lr':>9}{'wd':>7}{'ls':>6}{'mix':>6}"
          f"{'ema':>7}{'loss':>8}")
    for r in runs:
        print(f"{r['run']:<26}{r['aug']:>10}{r['input']:>7}{r['lr']:>9.1e}{r['wd']:>7.3f}"
              f"{r['ls']:>6.2f}{r['mix']:>6.2f}{r['ema']:>7.3f}{r['loss']:>8}")


if __name__ == '__main__':
    main()

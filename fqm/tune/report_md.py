"""生成最终结果 Markdown（多种子 + TTA），写入 fqm/最终结果.md。

用法（仓库根目录）：
    python fqm/tune/report_md.py --root fqm/runs/final --out fqm/最终结果.md
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


def collect(root: Path):
    rows = []
    for path in sorted(root.glob('*/results.json')):
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        tta = None
        tta_path = path.parent / 'tta_results.json'
        if tta_path.exists():
            with open(tta_path, encoding='utf-8') as f:
                tta = json.load(f)
        cfg = data['config']
        rows.append({
            'run': data['run_name'],
            'seed': cfg['train']['seed'],
            'acc': data['val_orig']['accuracy'],
            'auroc': data['val_orig']['auroc'],
            'acc_thr': data['val_orig'].get('accuracy_best_threshold'),
            'cm': data['val_orig']['confusion_matrix'],
            'r_real': data['val_orig']['recall_real'],
            'r_fake': data['val_orig']['recall_fake'],
            'jpeg_acc': data['val_jpeg_q75']['accuracy'],
            'jpeg_auroc': data['val_jpeg_q75']['auroc'],
            'tta_acc': (tta or {}).get('orig', {}).get('accuracy'),
            'tta_auroc': (tta or {}).get('orig', {}).get('auroc'),
            'tta_jpeg_acc': (tta or {}).get('jpeg', {}).get('accuracy'),
            'tta_jpeg_auroc': (tta or {}).get('jpeg', {}).get('auroc'),
            'best_epoch': data.get('best_epoch'),
            'params': data.get('params_total', 0),
            'peak': data.get('peak_memory_mb', 0),
            'minutes': data.get('train_minutes', 0.0),
            'epoch_seconds': data.get('epoch_train_seconds_mean', 0.0),
            'cfg': cfg,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='生成最终结果 Markdown')
    parser.add_argument('--root', type=str, default='fqm/runs/final')
    parser.add_argument('--prefix', type=str, default='', help='只汇总 run 名以此开头的运行')
    parser.add_argument('--out', type=str, default='fqm/最终结果.md')
    args = parser.parse_args()

    rows = collect(REPO / args.root)
    rows = [r for r in rows if r['run'].startswith(args.prefix)]
    if not rows:
        raise SystemExit('没有最终结果')

    cfg = rows[0]['cfg']
    data_cfg, train_cfg = cfg['data'], cfg['train']
    aug = data_cfg.get('aug', {})
    geometry = ('原尺度裁剪（仅在短边不足时上采样）' if aug.get('mode') == 'native'
                else '短边缩放后裁剪')
    lines = [
        '# fqm · 最终结果（原尺度裁剪 320 + 60 epoch + EMA）',
        '',
        '配置来自 [调优记录](调优记录.md) 第三轮的最优项，随机种子 42 / 17 / 43 / 97 各训练一次；',
        '每个 checkpoint 额外做五裁剪 + 水平翻转的 TTA（`fqm/evaluate_tta.py`）。',
        '所有模型从随机初始化训练，未加载任何预训练权重；固定划分与统一 JPEG 评测协议不变。',
        '',
        '## 配置',
        '',
        '| 项目 | 取值 |',
        '| --- | --- |',
        f"| 模型 | EfficientViM-M1，二分类头，参数量 {rows[0]['params'] / 1e6:.3f}M |",
        f"| 输入 | {geometry}，裁剪 {data_cfg['input_size']}×{data_cfg['input_size']}，"
        f"水平翻转 p={aug.get('hflip')} |",
        f"| 优化 | AdamW，lr {train_cfg['lr']:g}（前 {train_cfg['warmup_updates']} 步 3e-6→"
        f"{train_cfg['lr']:g}，其余余弦到 {train_cfg['lr_final']:g}），"
        f"wd {train_cfg['weight_decay']}，label smoothing {train_cfg['label_smoothing']} |",
        f"| 批量与预算 | 微批量 {train_cfg['micro_batch']} × 累积 {train_cfg['accum_steps']} = "
        f"有效批量 {train_cfg['micro_batch'] * train_cfg['accum_steps']}，"
        f"{train_cfg['epochs']} epoch × {train_cfg['updates_per_epoch']} 次更新 = "
        f"{train_cfg['epochs'] * train_cfg['updates_per_epoch']} 次 |",
        f"| 精度 / EMA | FP16 AMP + GradScaler；EMA decay {train_cfg.get('ema_decay', 0)} |",
        '| 选模 | 原图验证 AUROC 最高的 epoch（EMA 权重） |',
        '',
        '## 每个种子',
        '',
        '| run | 种子 | 最佳 epoch | 原图 acc | 原图 AUROC | 真实召回 | 生成召回 | 原图混淆矩阵 '
        '| JPEG acc | JPEG AUROC | TTA acc | TTA AUROC |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for r in rows:
        lines.append(
            f"| `{r['run']}` | {r['seed']} | {r['best_epoch']} | {r['acc']:.4f} | {r['auroc']:.4f} | "
            f"{r['r_real']:.4f} | {r['r_fake']:.4f} | {r['cm']} | {r['jpeg_acc']:.4f} | "
            f"{r['jpeg_auroc']:.4f} | {r['tta_acc']:.4f} | {r['tta_auroc']:.4f} |")

    lines += ['', '## 均值 ± 样本标准差', '', '| 指标 | 取值 |', '| --- | --- |']
    for key, name in (('acc', '原图准确率'), ('auroc', '原图 AUROC'),
                      ('r_real', '原图真实类召回'), ('r_fake', '原图生成类召回'),
                      ('jpeg_acc', 'JPEG 版准确率'), ('jpeg_auroc', 'JPEG 版 AUROC'),
                      ('tta_acc', 'TTA 原图准确率'), ('tta_auroc', 'TTA 原图 AUROC'),
                      ('tta_jpeg_acc', 'TTA JPEG 准确率'), ('tta_jpeg_auroc', 'TTA JPEG AUROC')):
        lines.append(f'| {name} | {mean_std([r[key] for r in rows])} |')
    lines.append(f"| 每轮训练耗时 | {mean_std([r['epoch_seconds'] for r in rows])} s |")
    lines.append(f"| 总训练时长 | {mean_std([r['minutes'] for r in rows])} 分钟 |")
    lines.append(f"| 峰值显存 | {max(r['peak'] for r in rows):.0f} MB |")

    lines += [
        '',
        '## 文件',
        '',
        '- 配置：`fqm/configs/efficientvim_best.json` 与 `fqm/tune/configs/final_*_seed*.json`',
        f"- 最优 checkpoint：`{args.root}/{rows[0]['run']}/best.pt`",
        f"- 逐轮日志与汇总：`{args.root}/<run>/log.jsonl`、`results.json`、`val_predictions_*.csv`",
        f"- TTA 结果：`{args.root}/<run>/tta_results.json`",
        '',
    ]

    out_path = REPO / args.out
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'写入 {out_path}（{len(rows)} 个种子）')


if __name__ == '__main__':
    main()

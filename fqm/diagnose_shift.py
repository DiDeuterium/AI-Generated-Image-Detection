"""诊断：在「先缩放再 q75 重编码」的验证集上评测，并导出测试集概率。

用法：python fqm/diagnose_shift.py --config <cfg> --checkpoint <pt> --scale 0.75
"""
import argparse, csv, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssd.data import build_loader, build_transform, load_norm_stats, read_splits, read_test_ids
from ssd.engine import compute_metrics, load_checkpoint, predict, resolve_amp
from train import build_model

REPO = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--scale', type=float, default=0.75)
    ap.add_argument('--quality', type=int, default=75)
    ap.add_argument('--num-workers', type=int, default=2)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    d = cfg['data']
    stats = load_norm_stats(REPO / d['norm_stats'])
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp, amp_dtype = resolve_amp(bool(cfg['train']['amp']), cfg['train']['amp_dtype'], dev)

    _, rows = read_splits(REPO / d['splits'], REPO / d['image_dir'])
    cache = REPO / f"fqm/cache/val_s{int(args.scale*100)}_q{args.quality}"
    cache.mkdir(parents=True, exist_ok=True)
    def make(row):
        dst = cache / (Path(row['image_id']).stem + '.jpg')
        if not dst.exists():
            with Image.open(row['path']) as im:
                im = im.convert('RGB')
                w, h = im.size
                im = im.resize((max(8, int(round(w*args.scale))), max(8, int(round(h*args.scale)))), Image.BICUBIC)
                im.save(dst, 'JPEG', quality=args.quality, subsampling=2)
        return {**row, 'path': dst}
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(make, rows))

    tf = build_transform(d['input_size'], d['resize_size'], stats['mean'], stats['std'], train=False, aug=d.get('aug', {}))
    loader = build_loader(rows, tf, 64, False, args.num_workers, 0)
    model = build_model(cfg['model']).to(dev)
    load_checkpoint(args.checkpoint, model, dev)
    probs = predict(model, loader, dev, amp, amp_dtype)
    labels = np.array([r['label'] for r in rows])
    m = compute_metrics(labels, probs)
    print(f"[shift] scale {args.scale} q{args.quality}: acc {m['accuracy']:.4f} auroc {m['auroc']:.4f} "
          f"R0 {m['recall_real']:.4f} R1 {m['recall_fake']:.4f}", flush=True)

    template = REPO / 'ai-generated-image-detection-benchmark' / 'sample_submission.csv'
    test_rows = read_test_ids(template, REPO / d['image_dir'] / 'dataset' / 'test')
    tprobs = predict(model, build_loader(test_rows, tf, 64, False, args.num_workers, 0), dev, amp, amp_dtype)
    out = REPO / 'fqm/runs/diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.checkpoint).parent.name
    with open(out / f'{name}_test_prob.csv', 'w', newline='') as f:
        wr = csv.writer(f); wr.writerow(['image_id', 'prob_fake'])
        for r, p in zip(test_rows, tprobs[:, 1]):
            wr.writerow([r['image_id'], f'{p:.6f}'])
    json.dump({'shift_metrics': m, 'scale': args.scale, 'quality': args.quality},
              open(out / f'{name}_shift.json', 'w'), indent=2)
    print('[shift] 写入', out / f'{name}_test_prob.csv')

if __name__ == '__main__':
    main()

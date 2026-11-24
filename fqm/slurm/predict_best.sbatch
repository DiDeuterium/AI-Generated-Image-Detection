#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/predict_%j.out
#SBATCH --error=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/predict_%j.err

# 用最终配置 + 指定 checkpoint 对测试集推理，导出比赛模板格式的提交文件并校验。
# 用法：sbatch fqm/slurm/predict_best.sbatch [config] [checkpoint]

set -euo pipefail

REPO=/remote-home1/qmfan/AI-Generated-Image-Detection
CONFIG=${1:-fqm/configs/efficientvim_best.json}
CKPT=${2:-fqm/runs/final/final_w5_cap60_seed42/best.pt}

source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
cd "$REPO"
export PYTHONUNBUFFERED=1

echo "[predict] $CONFIG + $CKPT on $(hostname)"
python fqm/predict.py --config "$CONFIG" --checkpoint "$CKPT"
python - "$CKPT" <<'EOF'
import csv
import sys
from pathlib import Path
sub = Path(sys.argv[1]).parent / 'submission.csv'
rows = list(csv.DictReader(open(sub)))
template = list(csv.DictReader(open('ai-generated-image-detection-benchmark/sample_submission.csv')))
assert [r['image_id'] for r in rows] == [r['image_id'] for r in template], '提交顺序与模板不一致'
assert len(rows) == len(template) == 1000
assert len({r['image_id'] for r in rows}) == 1000
assert all(r['label'] in ('0', '1') for r in rows)
print('[predict] 校验通过：1000 行、ID 顺序与模板一致、标签 0/1，AI 生成预测数',
      sum(int(r['label']) for r in rows))
EOF

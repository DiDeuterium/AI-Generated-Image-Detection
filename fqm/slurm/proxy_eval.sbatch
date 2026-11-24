#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/proxy_%j.out
#SBATCH --error=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/proxy_%j.err

# 多质量 JPEG 代理评测（对齐测试集编码），可一次评多个 checkpoint。
# 用法：sbatch fqm/slurm/proxy_eval.sbatch <config.json> <ckpt1> [ckpt2 ...]

set -euo pipefail
REPO=/remote-home1/qmfan/AI-Generated-Image-Detection
CFG="${1:?需要配置}"
shift
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
cd "$REPO"
export PYTHONUNBUFFERED=1
for ck in "$@"; do
  echo "[proxy] ==== $ck"
  python fqm/evaluate_proxy.py --config "$CFG" --checkpoint "$ck" --qualities 60 75 90
  python fqm/evaluate_proxy.py --config "$CFG" --checkpoint "$ck" --qualities 60 75 90 --tta
 done
echo "[proxy] 完成"

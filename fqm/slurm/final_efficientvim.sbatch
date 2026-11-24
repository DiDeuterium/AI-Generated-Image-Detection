#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/final_%A_%a.out
#SBATCH --error=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/final_%A_%a.err

# 最终配置复跑：训练 → 用最佳 checkpoint 做五裁剪+翻转 TTA 评测。
# 用法：sbatch --array=0-3 fqm/slurm/final_efficientvim.sbatch fqm/tune/manifest_final.txt

set -euo pipefail

MANIFEST="${1:?需要配置清单，例如 fqm/tune/manifest_final.txt}"
REPO=/remote-home1/qmfan/AI-Generated-Image-Detection

source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
cd "$REPO"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [[ -z "$CFG" ]]; then
  echo "[final] 清单没有第 $((SLURM_ARRAY_TASK_ID + 1)) 行"
  exit 0
fi

RUN_NAME=$(python -c "import json,sys; print(json.load(open('$CFG'))['output']['run_name'])")
echo "[final] task ${SLURM_ARRAY_TASK_ID} -> $CFG ($RUN_NAME) on $(hostname)"

python fqm/train.py --config "$CFG"
python fqm/evaluate_tta.py --config "$CFG" \
    --checkpoint "fqm/runs/final/$RUN_NAME/best.pt"
echo "[final] task ${SLURM_ARRAY_TASK_ID} 完成"

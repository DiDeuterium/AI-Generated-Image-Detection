#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/tune_%A_%a.out
#SBATCH --error=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/tune_%A_%a.err

# 调参阵列作业：每个任务跑清单里的一行配置（1 GPU / 任务）。
# 用法：sbatch fqm/slurm/tune_efficientvim.sbatch fqm/tune/manifest_w1.txt

set -euo pipefail

MANIFEST="${1:?需要配置文件清单，例如 fqm/tune/manifest_w1.txt}"
REPO=/remote-home1/qmfan/AI-Generated-Image-Detection

source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
cd "$REPO"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [[ -z "$CFG" ]]; then
  echo "[tune] 清单 $MANIFEST 没有第 $((SLURM_ARRAY_TASK_ID + 1)) 行"
  exit 0
fi

echo "[tune] task ${SLURM_ARRAY_TASK_ID} -> $CFG on $(hostname)"
python fqm/train.py --config "$CFG"
echo "[tune] task ${SLURM_ARRAY_TASK_ID} 完成"

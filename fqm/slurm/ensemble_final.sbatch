#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/ensemble_%j.out
#SBATCH --error=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/ensemble_%j.err

# 4 种子集成评测（可叠加 TTA）：报告 + 逐种子汇总 + 概率平均后的指标。
# 用法：sbatch fqm/slurm/ensemble_final.sbatch <config.json> <ckpt1> <ckpt2> <ckpt3> <ckpt4>

set -euo pipefail

REPO=/remote-home1/qmfan/AI-Generated-Image-Detection
CFG="${1:?需要配置路径}"
shift

source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
cd "$REPO"
export PYTHONUNBUFFERED=1

echo "[ens] config $CFG"
echo "[ens] checkpoints $*"
python fqm/tune/report_md.py
python fqm/tune/aggregate_final.py --root fqm/runs/final | tee fqm/logs/final_report.txt
python fqm/evaluate_ensemble.py --config "$CFG" --checkpoints "$@" --tta \
    --output fqm/runs/final/ensemble_tta_results.json
echo "[ens] 完成"

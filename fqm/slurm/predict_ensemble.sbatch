#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/enspred_%j.out
#SBATCH --error=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/enspred_%j.err

# 多 checkpoint（+TTA）测试集推理并导出提交文件。
# 用法：sbatch fqm/slurm/predict_ensemble.sbatch <config.json> <out.csv> <ckpt...>

set -euo pipefail

REPO=/remote-home1/qmfan/AI-Generated-Image-Detection
CFG="${1:?需要配置路径}"
OUT="${2:?需要输出 csv 路径}"
shift 2

source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
cd "$REPO"
export PYTHONUNBUFFERED=1

echo "[ens-pred] config $CFG -> $OUT"
python fqm/predict_ensemble.py --config "$CFG" --checkpoints "$@" --tta --output "$OUT"
echo "[ens-pred] 完成"

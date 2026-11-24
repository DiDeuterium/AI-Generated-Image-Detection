#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/%x_%j.out
#SBATCH --error=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/%x_%j.err

# EfficientViM-M1 二分类基线：先自检（前向/反向/更新/checkpoint 复现），再正式训练。
# 提交方式（仓库根目录）：
#   sbatch fqm/slurm/train_efficientvim_m1.sbatch
# 追加参数会透传给 fqm/train.py（例如 --seed 17）。

set -euo pipefail

REPO=/remote-home1/qmfan/AI-Generated-Image-Detection
CONFIG=fqm/configs/efficientvim_m1.json

source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
cd "$REPO"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p fqm/logs fqm/runs

echo "[job] $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname), partition ${SLURM_JOB_PARTITION}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -c "import torch; print('[env] torch', torch.__version__, '| cuda', torch.version.cuda, '| cudnn', torch.backends.cudnn.version())"

echo "[step 1/2] 开训自检"
python fqm/smoke_test.py --config "$CONFIG" --batch-size 8

echo "[step 2/2] 正式训练"
python fqm/train.py --config "$CONFIG" "$@"

echo "[job] 完成"

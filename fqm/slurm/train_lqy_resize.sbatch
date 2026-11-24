#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=fqm/logs/train_lqy_resize_%j.out
#SBATCH --error=fqm/logs/train_lqy_resize_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"
python -u fqm/lqy_resize_retrain.py --output fqm/runs/lqy_resize_20261122 \
  --csv fqm/submissions/exp-lqy-resize-20261122.csv

#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=01:30:00
#SBATCH --output=fqm/logs/native_texture_four_views_%j.out
#SBATCH --error=fqm/logs/native_texture_four_views_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"
python -u fqm/native_texture_base.py --output fqm/runs/native_texture_four_views_20261123 \
  --epochs 24 --view-pairs 2

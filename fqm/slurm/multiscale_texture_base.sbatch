#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=fqm/logs/multiscale_texture_base_%j.out
#SBATCH --error=fqm/logs/multiscale_texture_base_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"
python -u fqm/multiscale_texture_base.py --output fqm/runs/multiscale_texture_base_20261123 --epochs 24

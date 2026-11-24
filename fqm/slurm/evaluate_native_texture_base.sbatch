#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=fqm/logs/evaluate_native_texture_base_%j.out
#SBATCH --error=fqm/logs/evaluate_native_texture_base_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"
if (( $# == 0 )); then
  set -- --output fqm/runs/native_texture_base_eval_20261122
fi
python -u fqm/evaluate_native_texture_base.py "$@"

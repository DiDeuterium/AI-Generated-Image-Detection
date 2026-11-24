#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=fqm/logs/lqy_specialist_gate_%j.out
#SBATCH --error=fqm/logs/lqy_specialist_gate_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"
python -u fqm/lqy_specialist_gate.py --output fqm/runs/lqy_gate_20261122 \
  --candidate fqm/submissions/exp-lqy-gate-20261122.csv

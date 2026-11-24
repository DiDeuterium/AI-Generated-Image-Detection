#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=fqm/logs/reselect_texture_checkpoint_%j.out
#SBATCH --error=fqm/logs/reselect_texture_checkpoint_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"
python -u fqm/reselect_texture_checkpoint.py \
  --output "fqm/runs/reselect_texture_checkpoint_${SLURM_JOB_ID}" \
  --candidate "fqm/submissions/exp-reselected-texture-${SLURM_JOB_ID}.csv"

#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=fqm/logs/momentum_contrastive_texture_%j.out
#SBATCH --error=fqm/logs/momentum_contrastive_texture_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"

run="fqm/runs/momentum_contrastive_texture_${SLURM_JOB_ID}"
python -u fqm/contrastive_texture_base.py --output "$run" --pairing momentum
python -u fqm/evaluate_native_texture_base.py \
  --output "$run/evaluation" --trained "$run" --training-job "$SLURM_JOB_ID" \
  --reference-evaluation fqm/runs/self_supervised_autocorrelation_texture_7169/evaluation --baseline-online 0.727 \
  --comparison-evaluation fqm/runs/contrastive_texture_base_7192/evaluation \
  --candidate "fqm/submissions/exp-momentum-contrastive-texture-${SLURM_JOB_ID}.csv"

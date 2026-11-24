#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=fqm/logs/self_supervised_autocorrelation_texture_%j.out
#SBATCH --error=fqm/logs/self_supervised_autocorrelation_texture_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"

run="fqm/runs/self_supervised_autocorrelation_texture_${SLURM_JOB_ID}"
python -u fqm/self_supervised_texture_base.py --output "$run" --target autocorrelation
python -u fqm/evaluate_native_texture_base.py \
  --output "$run/evaluation" --trained "$run" --training-job "$SLURM_JOB_ID" \
  --reference-evaluation fqm/runs/native_texture_base_eval_20261122 --baseline-online 0.705 \
  --comparison-evaluation fqm/runs/self_supervised_texture_base_7164/evaluation \
  --candidate "fqm/submissions/exp-self-supervised-autocorrelation-${SLURM_JOB_ID}.csv"

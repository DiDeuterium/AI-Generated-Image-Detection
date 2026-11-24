#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=fqm/logs/deep_native_texture_base_%j.out
#SBATCH --error=fqm/logs/deep_native_texture_base_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"

run="fqm/runs/deep_native_texture_base_${SLURM_JOB_ID}"
python -u fqm/native_texture_base.py --output "$run" --epochs 24 --deep
python -u fqm/evaluate_native_texture_base.py \
  --output "$run/evaluation" --trained "$run" --training-job "$SLURM_JOB_ID" \
  --reference-evaluation fqm/runs/native_texture_base_eval_20261122 --baseline-online 0.705 \
  --comparison-evaluation fqm/runs/wide_native_texture_base_7151/evaluation \
  --candidate "fqm/submissions/exp-deep-native-texture-${SLURM_JOB_ID}.csv"

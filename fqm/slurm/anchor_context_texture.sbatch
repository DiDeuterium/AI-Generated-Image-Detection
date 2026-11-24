#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=fqm/logs/anchor_context_texture_%j.out
#SBATCH --error=fqm/logs/anchor_context_texture_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"

run="fqm/runs/anchor_context_texture_${SLURM_JOB_ID}"
baseline="fqm/runs/self_supervised_autocorrelation_texture_7169"
python -u fqm/patch_evidence_texture.py --output "$run" --anchor-context-only
python -u fqm/evaluate_native_texture_base.py \
  --output "$run/anchor_context/evaluation" --trained "$run/anchor_context" --training-job "$SLURM_JOB_ID" \
  --reference-evaluation "$baseline/evaluation" --baseline-online 0.727 --extra-probe-reference "$baseline" \
  --comparison-evaluation fqm/runs/patch_evidence_texture_7198/patch_evidence/evaluation \
  --candidate "fqm/submissions/exp-anchor-context-texture-${SLURM_JOB_ID}.csv"

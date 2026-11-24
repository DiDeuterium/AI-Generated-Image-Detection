#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=fqm/logs/balanced_anchor_context_texture_%j.out
#SBATCH --error=fqm/logs/balanced_anchor_context_texture_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"

run="fqm/runs/balanced_anchor_context_texture_${SLURM_JOB_ID}"
baseline="fqm/runs/anchor_context_texture_7200/anchor_context"
epochs=({1..24})
python -u fqm/evaluate_native_texture_base.py \
  --output "$run/reselect_7200" --trained "$baseline" --training-job 7200 --checkpoint-epochs "${epochs[@]}" \
  --reference-evaluation "$baseline/evaluation" --baseline-online 0.769 --extra-probe-reference "$baseline" \
  --candidate "fqm/submissions/exp-reselected-anchor-context-${SLURM_JOB_ID}.csv"
python -u fqm/patch_evidence_texture.py --output "$run/balanced" --anchor-context-only --balanced-acquisition
python -u fqm/evaluate_native_texture_base.py \
  --output "$run/balanced/anchor_context/evaluation" --trained "$run/balanced/anchor_context" --training-job "$SLURM_JOB_ID" \
  --checkpoint-epochs "${epochs[@]}" --comparison-evaluation "$run/reselect_7200" \
  --reference-evaluation "$baseline/evaluation" --baseline-online 0.769 --extra-probe-reference "$baseline" \
  --candidate "fqm/submissions/exp-balanced-anchor-context-${SLURM_JOB_ID}.csv"

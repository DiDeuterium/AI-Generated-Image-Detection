#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=fqm/logs/multiband_anchor_context_texture_%j.out
#SBATCH --error=fqm/logs/multiband_anchor_context_texture_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"

run="fqm/runs/multiband_anchor_context_texture_${SLURM_JOB_ID}"
baseline="fqm/runs/anchor_context_texture_7200/anchor_context"
reference="fqm/runs/balanced_anchor_context_texture_7202/reselect_7200"
epochs=({1..24})
python -u fqm/patch_evidence_texture.py --output "$run" --anchor-context-only --multiband
python -u fqm/evaluate_native_texture_base.py \
  --output "$run/anchor_context/evaluation" --trained "$run/anchor_context" --training-job "$SLURM_JOB_ID" \
  --checkpoint-epochs "${epochs[@]}" --reference-evaluation "$reference" --baseline-online 0.787 \
  --extra-probe-reference "$baseline" \
  --candidate "fqm/submissions/exp-multiband-anchor-context-${SLURM_JOB_ID}.csv"

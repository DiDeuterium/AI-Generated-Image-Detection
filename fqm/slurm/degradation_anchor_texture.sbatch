#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=fqm/logs/degradation_anchor_texture_%j.out
#SBATCH --error=fqm/logs/degradation_anchor_texture_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"

run="fqm/runs/degradation_anchor_texture_${SLURM_JOB_ID}"
baseline="fqm/runs/anchor_context_texture_7200/anchor_context"
reference="fqm/runs/balanced_anchor_context_texture_7202/reselect_7200"
epochs=({1..24})
python -u fqm/degradation_anchor_texture.py --output "$run"
python -u fqm/evaluate_native_texture_base.py \
  --output "$run/evaluation" --trained "$run" --training-job "$SLURM_JOB_ID" \
  --checkpoint-epochs "${epochs[@]}" --reference-evaluation "$reference" --baseline-online 0.787 \
  --extra-probe-reference "$baseline" \
  --comparison-evaluation fqm/runs/balanced_anchor_context_texture_7202/balanced/anchor_context/evaluation \
  --candidate "fqm/submissions/exp-degradation-anchor-texture-${SLURM_JOB_ID}.csv"

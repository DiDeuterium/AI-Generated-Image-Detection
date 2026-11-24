#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=fqm/logs/patch_evidence_texture_%j.out
#SBATCH --error=fqm/logs/patch_evidence_texture_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"

run="fqm/runs/patch_evidence_texture_${SLURM_JOB_ID}"
baseline="fqm/runs/self_supervised_autocorrelation_texture_7169"
python -u fqm/patch_evidence_texture.py --output "$run"
python -u fqm/evaluate_native_texture_base.py \
  --output "$run/grid_control/evaluation" --trained "$run/grid_control" --training-job "$SLURM_JOB_ID" \
  --reference-evaluation "$baseline/evaluation" --baseline-online 0.727 --extra-probe-reference "$baseline"
python -u fqm/evaluate_native_texture_base.py \
  --output "$run/patch_evidence/evaluation" --trained "$run/patch_evidence" --training-job "$SLURM_JOB_ID" \
  --reference-evaluation "$baseline/evaluation" --baseline-online 0.727 --extra-probe-reference "$baseline" \
  --comparison-evaluation "$run/grid_control/evaluation" \
  --candidate "fqm/submissions/exp-patch-evidence-texture-${SLURM_JOB_ID}.csv"

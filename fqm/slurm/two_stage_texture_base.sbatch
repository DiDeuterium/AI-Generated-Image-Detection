#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=fqm/logs/two_stage_texture_base_%j.out
#SBATCH --error=fqm/logs/two_stage_texture_base_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"

run="fqm/runs/two_stage_texture_base_${SLURM_JOB_ID}"
python -u fqm/two_stage_texture_base.py --output "$run"
for arm in frozen finetuned; do
  python -u fqm/evaluate_native_texture_base.py \
    --output "$run/$arm/evaluation" --trained "$run/$arm" --training-job "$SLURM_JOB_ID" \
    --checkpoint-epochs 6 12 \
    --reference-evaluation fqm/runs/self_supervised_autocorrelation_texture_7169/evaluation --baseline-online 0.727 \
    --candidate "fqm/submissions/exp-two-stage-texture-${arm}-${SLURM_JOB_ID}.csv"
done

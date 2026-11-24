#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=fqm/logs/train_target_smoothing_%j.out
#SBATCH --error=fqm/logs/train_target_smoothing_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
trial=target_hybrid_smooth_20261121
control=target_hybrid_capped_20261121
output=fqm/runs/target_smoothing_20261121
test -s "fqm/runs/tune/$control/best.pt"
test ! -e "$output"
python -u fqm/train.py --config "fqm/tune/configs/$trial.json"
python -u fqm/audit_target_profile.py --profiles fqm/runs/target_profile_20261121/profiles.json \
    --output "$output" --heldout "$control" "$trial" \
    --cases orig geometry encoding both legacy_strong both_blur
python -u fqm/compare_target_profile.py "$output" "$control" "$trial"

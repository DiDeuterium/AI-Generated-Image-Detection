#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=fqm/logs/ensemble_views_%j.out
#SBATCH --error=fqm/logs/ensemble_views_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
root=fqm/runs/ensemble_views_20261121
output=fqm/submissions/exp-7047-7048_ensemble.csv
test ! -e "$root" && test ! -e "$output"
args=(--profiles fqm/runs/target_profile_20261121/profiles.json target_control_20261121 target_matched_20261121)
python -u fqm/audit_target_profile.py --output "$root/center" "${args[@]}"
python -u fqm/audit_target_profile.py --output "$root/five" --five-crop "${args[@]}"
choice=$(python -u fqm/evaluate_ensemble_views.py "$root")
echo "Chosen inference: $choice"
if [ "$choice" = none ]; then exit 0; fi
tta=()
if [ "$choice" = five ]; then tta=(--tta --no-flip); fi
python -u fqm/predict_ensemble.py --config fqm/tune/configs/target_matched_20261121.json \
  --checkpoints fqm/runs/tune/target_control_20261121/best.pt fqm/runs/tune/target_matched_20261121/best.pt \
  --output "$output" --batch-size 8 --num-workers 2 "${tta[@]}"

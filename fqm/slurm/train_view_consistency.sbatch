#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=fqm/logs/train_view_consistency_%j.out
#SBATCH --error=fqm/logs/train_view_consistency_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
if [[ "${1:?specify the experiment or compare}" == compare ]]; then
  shift
  output="${1:?specify the comparison output directory}"
  shift
  python -u fqm/audit_target_profile.py --profiles fqm/runs/target_profile_20261121/profiles.json \
    --output "$output" --heldout "$@"
  python -u fqm/compare_target_profile.py "$output" "$@"
else
  python -u fqm/train.py --config "fqm/tune/configs/$1.json"
fi

#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=fqm/logs/train_target_profile_%j.out
#SBATCH --error=fqm/logs/train_target_profile_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
name="${1:?Specify the run/config name}"
root=fqm/runs/target_profile_20261121
python -u fqm/train.py --config "fqm/tune/configs/$name.json"
python -u fqm/audit_target_profile.py --profiles "$root/profiles.json" \
  --output "$root/$name" --heldout "$name"

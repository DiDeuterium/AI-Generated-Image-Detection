#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=fqm/logs/audit_proxy_%j.out
#SBATCH --error=fqm/logs/audit_proxy_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python -u fqm/audit_proxy_shift.py "$@"

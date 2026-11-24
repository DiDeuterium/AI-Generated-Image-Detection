#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --output=fqm/logs/audit_width500_%j.out
#SBATCH --error=fqm/logs/audit_width500_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python -u fqm/audit_width500.py "$@"

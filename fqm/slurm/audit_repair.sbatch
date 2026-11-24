#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=fqm/logs/audit_repair_%j.out
#SBATCH --error=fqm/logs/audit_repair_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
python -u fqm/train.py --config fqm/tune/configs/audit_repair_nocap_20261121.json
python -u fqm/audit_proxy_shift.py --output fqm/runs/audit_repair_eval_20261121 audit_repair_nocap_20261121

#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/diag_%j.out
#SBATCH --error=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/diag_%j.err
set -euo pipefail
REPO=/remote-home1/qmfan/AI-Generated-Image-Detection
cd "$REPO"
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONUNBUFFERED=1
python fqm/diagnose_shift.py --config fqm/tune/configs/w6_cap60_jpegaug.json --checkpoint fqm/runs/tune/w6_cap60_jpegaug/best.pt --scale 0.75
python fqm/diagnose_shift.py --config fqm/tune/configs/w6_cap60_jpegaug.json --checkpoint fqm/runs/tune/w6_cap60_jpegaug/best.pt --scale 0.6
echo done

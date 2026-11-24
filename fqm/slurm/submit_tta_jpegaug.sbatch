#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/tta_sub_%j.out
#SBATCH --error=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/tta_sub_%j.err
set -euo pipefail
REPO=/remote-home1/qmfan/AI-Generated-Image-Detection
cd "$REPO"
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONUNBUFFERED=1
echo "[tta-sub] start $(date)"
python fqm/ensemble_q75.py --config fqm/tune/configs/w6_cap60_jpegaug.json \
  --checkpoints fqm/runs/tune/w6_cap60_jpegaug/best.pt \
  --tta-scales 320 352 --qualities 75 \
  --submit fqm/runs/tune/w6_cap60_jpegaug/submission_tta.csv \
  --output fqm/runs/tune/w6_cap60_jpegaug/proxy_tta_scales.json
echo "[tta-sub] done $(date)"

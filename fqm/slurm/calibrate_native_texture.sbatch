#!/bin/bash
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=14G
#SBATCH --time=00:15:00
#SBATCH --output=fqm/logs/calibrate_native_texture_%j.out
#SBATCH --error=fqm/logs/calibrate_native_texture_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"
python -u fqm/calibrate_native_texture.py --output fqm/runs/native_texture_four_views_calibrated_20261123 \
  --candidate fqm/submissions/exp-native-texture-four-views-calibrated-20261123.csv

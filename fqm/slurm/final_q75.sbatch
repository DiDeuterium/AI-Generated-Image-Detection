#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/finalq75_%j.out
#SBATCH --error=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/finalq75_%j.err

# 收尾作业：按 q75 指标挑选最优变体 -> 多尺度 TTA 集成评测 -> 导出测试集提交。
# 提交方式：sbatch --dependency=afterany:<训练阵列1>:<阵列2>... fqm/slurm/final_q75.sbatch

set -euo pipefail

REPO=/remote-home1/qmfan/AI-Generated-Image-Detection
cd "$REPO"
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONUNBUFFERED=1

echo "[final-q75] start $(date)"
python fqm/tune/summarize.py --prefix w7_ | tee fqm/logs/summary_w7.txt
python fqm/tune/summarize.py --prefix w8_ | tee fqm/logs/summary_w8.txt
python fqm/tune/summarize.py --prefix w9_ | tee fqm/logs/summary_w9.txt || true
python fqm/tune/pick_q75.py --out fqm/logs/ensemble_list.txt

CFG=$(sed -n '1p' fqm/logs/ensemble_list.txt)
mapfile -t CKPTS < <(sed -n '2,$p' fqm/logs/ensemble_list.txt)
echo "[final-q75] config=$CFG models=${#CKPTS[@]}"

python fqm/ensemble_q75.py --config "$CFG" --checkpoints "${CKPTS[@]}" \
    --tta-scales 320 352 --qualities 60 75 90 \
    --submit fqm/runs/final/submission_q75_ensemble.csv \
    --output fqm/runs/final/ensemble_q75_results.json

python fqm/tune/pick_q75.py --out fqm/logs/ensemble_list_single.txt --max-models 1
CFG1=$(sed -n '1p' fqm/logs/ensemble_list_single.txt)
CKPT1=$(sed -n '2p' fqm/logs/ensemble_list_single.txt)
python fqm/evaluate_proxy.py --config "$CFG1" --checkpoint "$CKPT1" \
    --qualities 60 75 90 --tta-scales 320 352 | tee fqm/logs/single_q75_tta.txt
echo "[final-q75] done $(date)"

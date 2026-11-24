#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-a800
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/orch_%j.out
#SBATCH --error=/remote-home1/qmfan/AI-Generated-Image-Detection/fqm/logs/orch_%j.err

# 编排作业（不需要 GPU）：第五轮结束后汇总 -> 全局选最优 -> 提交 4 种子复跑
# -> 提交「汇总 + 4 种子 TTA 集成」作业（用 afterok 依赖串起来）。
# 提交方式：sbatch --dependency=afterany:<wave5 数组作业号> fqm/slurm/orchestrate_cap.sbatch

set -uo pipefail

REPO=/remote-home1/qmfan/AI-Generated-Image-Detection
PART=fnlp-3090,fnlp-a800
cd "$REPO"

source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONUNBUFFERED=1

echo "[orch] start $(date) on $(hostname)"

echo "[orch] 汇总第五轮"
python fqm/tune/summarize.py --prefix w5_ | tee fqm/logs/summary_w5.txt
echo "[orch] 全局选最优配置"
python fqm/tune/make_final.py | tee fqm/logs/make_final_cap.txt
BEST=$(sed -n 's/.*选中 \([A-Za-z0-9_]*\):.*/\1/p' fqm/logs/make_final_cap.txt | head -1)
if [[ -z "$BEST" ]]; then
  echo "[orch] 未能确定最优配置，退出"
  exit 1
fi
echo "[orch] 最优配置 = $BEST"

FINAL_JOB=$(sbatch --partition="$PART" --array=0-3 \
  fqm/slurm/final_efficientvim.sbatch fqm/tune/manifest_final.txt | awk '{print $4}')
echo "[orch] 提交最终复跑阵列 $FINAL_JOB"

CKPTS=""
for s in 42 17 43 97; do
  CKPTS="$CKPTS fqm/runs/final/final_${BEST}_seed${s}/best.pt"
done
CFG="fqm/tune/configs/final_${BEST}_seed42.json"

sbatch --partition="$PART" --job-name=test --gres=gpu:1 --cpus-per-task=6 --mem=24G \
  --time=01:00:00 --dependency="afterok:${FINAL_JOB}" \
  --output="$REPO/fqm/logs/ensemble_%j.out" \
  --error="$REPO/fqm/logs/ensemble_%j.err" \
  --wrap "cd $REPO && source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh && \
    conda activate CV && \
    python fqm/tune/report_md.py && \
    python fqm/tune/aggregate_final.py --root fqm/runs/final > fqm/logs/final_report.txt && \
    python fqm/evaluate_ensemble.py --config $CFG $CKPTS --tta \
      --output fqm/runs/final/ensemble_tta_results.json"

echo "[orch] 已提交集成评测（依赖 ${FINAL_JOB}），编排结束 $(date)"

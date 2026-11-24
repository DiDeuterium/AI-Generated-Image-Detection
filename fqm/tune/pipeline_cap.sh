#!/bin/bash
# 第五轮（CAP 模块）之后的自动流水线：
#   等 wave5 阵列 -> 汇总 -> 全局选最优（含 CAP 配置）-> 4 种子复跑 + TTA -> 4 种子集成评测
# 启动：nohup fqm/tune/pipeline_cap.sh &

set -uo pipefail

REPO=/remote-home1/qmfan/AI-Generated-Image-Detection
PY=/remote-home1/qmfan/miniconda3/envs/CV/bin/python
LOG=/tmp/pipeline_cap.log
PART=${PART:-fnlp-a800}
cd "$REPO"

log() { echo "[cap] $(date '+%H:%M:%S') $*" >> "$LOG"; }

log "等待第五轮阵列 6977（分区 $PART）"
while squeue -u qmfan -h | grep -q ' 6977'; do sleep 60; done
log '第五轮结束，汇总'

$PY fqm/tune/summarize.py --prefix w5_ > /tmp/summary_w5.txt 2>&1
$PY fqm/tune/make_final.py > /tmp/make_final_cap.txt 2>&1
cat /tmp/make_final_cap.txt >> "$LOG"

log '提交最终复跑（4 种子，训练 + TTA）'
JOBID=$(sbatch --partition="$PART" --array=0-3 fqm/slurm/final_efficientvim.sbatch \
        fqm/tune/manifest_final.txt | awk '{print $4}')
log "最终阵列 $JOBID"
while squeue -u qmfan -h | grep -q " $JOBID"; do sleep 60; done

log '汇总多种子结果'
$PY fqm/tune/report_md.py > /tmp/report_cap.log 2>&1
$PY fqm/tune/aggregate_final.py --root fqm/runs/final > /tmp/final_report.txt 2>&1

log '提交 4 种子集成（TTA）评测'
BEST=$(sed -n 's/.*选中 \(w[0-9a-z_]*\):.*/\1/p' /tmp/make_final_cap.txt)
CKPTS=""
for s in 42 17 43 97; do CKPTS="$CKPTS fqm/runs/final/final_${BEST}_seed${s}/best.pt"; done
CFG="fqm/tune/configs/final_${BEST}_seed42.json"
sbatch --partition="$PART" --job-name=test --gres=gpu:1 --cpus-per-task=6 --mem=24G \
  --time=01:00:00 \
  --output="$REPO/fqm/logs/ensemble_%j.out" --error="$REPO/fqm/logs/ensemble_%j.err" \
  --wrap "cd $REPO && source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh && \
    conda activate CV && python fqm/evaluate_ensemble.py --config $CFG $(
      for c in $CKPTS; do echo -n "--checkpoints $c "; done) --tta \
    --output fqm/runs/final/ensemble_tta_results.json" >> "$LOG" 2>&1
log '集成评测已提交，全部流程结束'

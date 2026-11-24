#!/bin/bash
# 调优流水线（后台运行）：等第三轮结束 -> 汇总 -> 选最优配置 -> 4 个种子复跑 + TTA -> 汇总
# 启动方式：nohup /remote-home1/qmfan/AI-Generated-Image-Detection/fqm/tune/pipeline.sh &

set -uo pipefail

REPO=/remote-home1/qmfan/AI-Generated-Image-Detection
PY=/remote-home1/qmfan/miniconda3/envs/CV/bin/python
LOG=/tmp/pipeline.log
cd "$REPO"

log() { echo "[pipeline] $(date '+%H:%M:%S') $*" >> "$LOG"; }

log '等待第三轮阵列作业 6954 结束'
while squeue -u qmfan -h | grep -q ' 6954'; do sleep 30; done
log '第三轮结束，开始汇总'

$PY fqm/tune/summarize.py > /tmp/summary_tune.txt 2>&1
$PY fqm/tune/make_final.py > /tmp/make_final.txt 2>&1
cat /tmp/make_final.txt >> "$LOG"
log '提交最终复跑阵列（4 个种子，训练 + TTA）'
JOBID=$(sbatch --array=0-3 fqm/slurm/final_efficientvim.sbatch fqm/tune/manifest_final.txt | awk '{print $4}')
log "最终阵列作业 $JOBID"

while squeue -u qmfan -h | grep -q " $JOBID"; do sleep 30; done
log '最终复跑结束，汇总多种子结果'

$PY fqm/tune/summarize.py --root fqm/runs/final --prefix final_ > /tmp/final_summary.txt 2>&1
$PY fqm/tune/aggregate_final.py --root fqm/runs/final > /tmp/final_report.txt 2>&1
log '全部完成'

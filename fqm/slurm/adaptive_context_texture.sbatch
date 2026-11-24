#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=fnlp-3090,fnlp-a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=fqm/logs/adaptive_context_texture_%j.out
#SBATCH --error=fqm/logs/adaptive_context_texture_%j.err
set -eo pipefail
source /remote-home1/qmfan/miniconda3/etc/profile.d/conda.sh
conda activate CV
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$PWD/fqm/runs/lqy_source_b23beec"

PYTHONPATH="$PWD/fqm:$PYTHONPATH" python -u - <<'PY'
import torch
from patch_evidence_texture import AdaptiveContextTexture

model = AdaptiveContextTexture().cuda().eval()
patches = torch.randn(2, 3, model.attention.in_features, device="cuda")
valid = torch.ones(2, 3, dtype=torch.bool, device="cuda")
anchors = torch.tensor([[True, True, True], [True, True, False]], device="cuda")
overlap = torch.eye(3, device="cuda").repeat(2, 1, 1)
with torch.no_grad():
    model.context_correction[-1].bias.fill_(100)
    logits, base = model.classify_with_context(patches, overlap, valid, anchors)
    torch.testing.assert_close(logits[0], base[0], atol=0, rtol=0)
    torch.testing.assert_close((logits - base)[1], torch.tensor([-.8, .8], device="cuda"))
print("Adaptive context check passed on the allocated GPU.")
PY

run="fqm/runs/adaptive_context_texture_${SLURM_JOB_ID}"
baseline="fqm/runs/anchor_context_texture_7200/anchor_context"
reference="fqm/runs/balanced_anchor_context_texture_7202/reselect_7200"
epochs=({1..24})
python -u fqm/patch_evidence_texture.py --output "$run" --anchor-context-only --adaptive-context
python -u fqm/evaluate_native_texture_base.py \
  --output "$run/anchor_context/evaluation" --trained "$run/anchor_context" --training-job "$SLURM_JOB_ID" \
  --checkpoint-epochs "${epochs[@]}" --reference-evaluation "$reference" --baseline-online 0.787 \
  --extra-probe-reference "$baseline" \
  --candidate "fqm/submissions/exp-adaptive-context-texture-${SLURM_JOB_ID}.csv"

#!/bin/bash
# Auto-eval script: runs eval on 1000/3000/4999 checkpoints (A2L)
# Usage: bash scripts/auto_eval_ssv2.sh

set -e

CHECKPOINT_BASE="checkpoints/pi0_fast_ssv2/ssv2_a2l_v1"
LABELS="ss-v2/labels/filtered_validation.json"
FRAMES="../datasets/20bn-something-something-v2-frames"
STEPS=(1000 3000 4999)

cd /home/main/storage/sungjin/openpi

echo "[auto_eval] Starting eval on checkpoints: ${STEPS[*]}"

for STEP in "${STEPS[@]}"; do
    CKPT="${CHECKPOINT_BASE}/${STEP}"
    if [ ! -d "$CKPT" ]; then
        echo "[auto_eval] Skipping step ${STEP}: checkpoint not found"
        continue
    fi
    echo ""
    echo "=========================================="
    echo "[auto_eval] Evaluating step ${STEP}..."
    echo "=========================================="
    uv run python scripts/eval_ssv2.py \
        --checkpoint "$CKPT" \
        --labels "$LABELS" \
        --frames "$FRAMES" \
        --num_samples 500 \
        --batch_size 8 \
        2>&1 | tee "logs/eval_ssv2_step${STEP}.log"
done

echo ""
echo "[auto_eval] All evals complete. Logs in logs/eval_ssv2_step*.log"

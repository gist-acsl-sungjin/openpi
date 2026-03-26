#!/usr/bin/env bash
# Evaluate libero_r_A and libero_r_C across multiple checkpoints in parallel.
#
# Usage:
#   bash scripts/eval_libero_all_steps.sh
#   STEPS="5000 10000" bash scripts/eval_libero_all_steps.sh
#
# A uses port 8000, C uses port 8001 (per eval_libero.sh convention).
# Each step: A and C are launched in parallel, then we wait before moving on.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

STEPS="${STEPS:-1000 5000 10000 15000 20000}"

echo "=== Batch LIBERO eval ==="
echo "Steps: $STEPS"
echo ""

for STEP in $STEPS; do
    echo "──────────────────────────────────────────"
    echo "Step $STEP: launching A and C in parallel"
    echo "──────────────────────────────────────────"

    bash "$SCRIPT_DIR/eval_libero.sh" A libero_r_A_v1 "$STEP" &
    PID_A=$!

    bash "$SCRIPT_DIR/eval_libero.sh" C libero_r_C_v1 "$STEP" &
    PID_C=$!

    # Wait for both to finish before next step
    wait $PID_A && echo "[Step $STEP] A done." || echo "[Step $STEP] A FAILED (exit $?)"
    wait $PID_C && echo "[Step $STEP] C done." || echo "[Step $STEP] C FAILED (exit $?)"

    echo ""
done

echo "=== All done. Summary ==="
echo ""
for STEP in $STEPS; do
    for VARIANT in A C; do
        EXP=$([ "$VARIANT" = "A" ] && echo "libero_r_A_v1" || echo "libero_r_C_v1")
        LOG="$REPO_ROOT/eval-logs/libero_${VARIANT}_${EXP}_step${STEP}/eval.log"
        if [[ -f "$LOG" ]]; then
            RESULT=$(grep -E "Total success rate|total success" "$LOG" | tail -1)
            printf "  [%s step%s] %s\n" "$VARIANT" "$STEP" "${RESULT:-'(no result line found)'}"
        else
            printf "  [%s step%s] log not found\n" "$VARIANT" "$STEP"
        fi
    done
done

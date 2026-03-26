#!/usr/bin/env bash
# Evaluate model C on "put both moka pots on the stove" until 20 warning-free episodes complete.
#
# Usage:
#   bash scripts/eval_moka_pots_C.sh [STEP]
#
# Default STEP: 24999

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VARIANT="C"
EXP_NAME="libero_r_C_v1"
STEP="${1:-24999}"
CLEAN_TRIALS=20  # run until this many warning-free episodes complete
TASK_ID=8  # put both moka pots on the stove

CONFIG="pi0_fast_libero_r_${VARIANT}"
CKPT_DIR="$REPO_ROOT/checkpoints/${CONFIG}/${EXP_NAME}/${STEP}"
PORT=8002

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT_DIR"
    exit 1
fi

VIDEO_OUT="$REPO_ROOT/eval-logs/libero_${VARIANT}_${EXP_NAME}_step${STEP}_moka_pots_${NUM_TRIALS}trials"
mkdir -p "$VIDEO_OUT"

PYTHON="$REPO_ROOT/.venv/bin/python"

echo ""
echo "=== Moka Pots Focused Eval: Variant ${VARIANT}, Step ${STEP} ==="
echo "  task_id   : $TASK_ID (put both moka pots on the stove)"
echo "  trials    : $NUM_TRIALS"
echo "  checkpoint: $CKPT_DIR"
echo "  port      : $PORT"
echo "  video_out : $VIDEO_OUT"
echo ""

# ── Start policy server ───────────────────────────────────────────────────────
echo "[1/2] Starting policy server on port $PORT ..."
"$PYTHON" scripts/serve_policy.py \
    --port "$PORT" \
    policy:checkpoint \
    --policy.config "$CONFIG" \
    --policy.dir "$CKPT_DIR" &
SERVER_PID=$!

trap 'echo "Stopping server (PID $SERVER_PID) ..."; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 60); do
    if "$PYTHON" -c "
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(('localhost', $PORT))
    s.close()
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null; then
        echo "Server ready."
        break
    fi
    sleep 2
done

# ── Run eval ─────────────────────────────────────────────────────────────────
echo "[2/2] Running focused eval on moka pots task ..."
MUJOCO_GL=osmesa \
LIBERO_CONFIG_PATH=/tmp/libero_config \
PYTHONPATH="$REPO_ROOT/third_party/libero" \
"$PYTHON" examples/libero/main.py \
    --args.host localhost \
    --args.port "$PORT" \
    --args.task-suite-name libero_10 \
    --args.clean-trials-per-task "$CLEAN_TRIALS" \
    --args.task-orders "$TASK_ID" \
    --args.video-out-path "$VIDEO_OUT" \
    2>&1 | tee "$VIDEO_OUT/eval.log"

echo ""
echo "=== Done. Results in $VIDEO_OUT/eval.log ==="
grep -E "Total success rate|clean success rate|task success rate" "$VIDEO_OUT/eval.log" | tail -5

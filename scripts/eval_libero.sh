#!/usr/bin/env bash
# Evaluate a trained pi0-FAST policy on LIBERO-10.
#
# Usage:
#   bash scripts/eval_libero.sh <VARIANT> <EXP_NAME> [STEP]
#
# Examples:
#   bash scripts/eval_libero.sh A libero_r_A_v1
#   bash scripts/eval_libero.sh C libero_r_C_v1 24999
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval_libero.sh A libero_r_A_v1
#
# Arguments:
#   VARIANT   : A or C (determines config name: pi0_fast_libero_r_A / pi0_fast_libero_r_C)
#   EXP_NAME  : experiment name (subfolder under checkpoints/pi0_fast_libero_r_<VARIANT>/)
#   STEP      : checkpoint step (default: latest)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${1:?Usage: $0 <VARIANT> <EXP_NAME> [STEP]}"
EXP_NAME="${2:?Usage: $0 <VARIANT> <EXP_NAME> [STEP]}"
STEP="${3:-}"

CONFIG="pi0_fast_libero_r_${VARIANT}"
CKPT_BASE="$REPO_ROOT/checkpoints/${CONFIG}/${EXP_NAME}"

# Resolve checkpoint step
if [[ -z "$STEP" ]]; then
    STEP=$(ls "$CKPT_BASE" | grep -E '^[0-9]+$' | sort -n | tail -1)
    echo "Using latest checkpoint step: $STEP"
fi
CKPT_DIR="$CKPT_BASE/$STEP"

if [[ ! -d "$CKPT_DIR" ]]; then
    echo "ERROR: Checkpoint not found: $CKPT_DIR"
    exit 1
fi

# Port: A→8000, B→8001, C→8002, D→8003 (allows parallel eval)
case "$VARIANT" in
  A) PORT=8000 ;;
  B) PORT=8001 ;;
  C) PORT=8002 ;;
  D) PORT=8003 ;;
  *) PORT=8001 ;;
esac

VIDEO_OUT="$REPO_ROOT/eval-logs/libero_${VARIANT}_${EXP_NAME}_step${STEP}"
mkdir -p "$VIDEO_OUT"

PYTHON="$REPO_ROOT/.venv/bin/python"

echo ""
echo "=== LIBERO Eval: Variant ${VARIANT} ==="
echo "  config    : $CONFIG"
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
    --policy.dir "$CKPT_DIR" &> "$VIDEO_OUT/server.log" &
SERVER_PID=$!

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
echo "[2/2] Running LIBERO-10 eval ..."
MUJOCO_GL=osmesa \
LIBERO_CONFIG_PATH=/tmp/libero_config \
PYTHONPATH="$REPO_ROOT/third_party/libero" \
"$PYTHON" examples/libero/main.py \
    --args.host localhost \
    --args.port "$PORT" \
    --args.task-suite-name libero_10 \
    --args.clean-trials-per-task 20 \
    --args.video-out-path "$VIDEO_OUT" \
    2>&1 | tee "$VIDEO_OUT/eval.log"

# ── Cleanup ──────────────────────────────────────────────────────────────────
echo "Stopping server (PID $SERVER_PID) ..."
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

echo ""
echo "=== Done. Results in $VIDEO_OUT/eval.log ==="
grep -E "Total success rate|total success" "$VIDEO_OUT/eval.log" | tail -3

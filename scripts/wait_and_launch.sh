#!/usr/bin/env bash
# Wait for Stage 1 (10k) and norm_stats to complete, then launch Stage 2 A+C.
#
# Usage:
#   bash scripts/wait_and_launch.sh [EXP_SUFFIX]
#
# Runs in the background (nohup-friendly). Launches:
#   GPU 0-3: Variant A  (base → libero-10-r)
#   GPU 4-7: Variant C  (Stage1-A2L → libero-10-r)
#
# Each variant gets its own tmux window in the current session.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXP_SUFFIX="${1:-v1}"
STAGE1_CKPT="$REPO_ROOT/checkpoints/pi0_fast_ssv2/ssv2_a2l_v1/9999/params"
NORM_STATS="$REPO_ROOT/assets/pi0_fast_libero_r_A/libero-10-r/norm_stats.json"
LOG="$REPO_ROOT/logs/wait_and_launch.log"
POLL_INTERVAL=600  # seconds between checks

mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== wait_and_launch started ==="
log "Waiting for:"
log "  Stage 1 ckpt : $STAGE1_CKPT"
log "  norm_stats   : $NORM_STATS"

# ── Wait for both prerequisites ───────────────────────────────────────────────
while true; do
    CKPT_OK=false
    NORM_OK=false

    # Stage 1 checkpoint: directory must exist and have no tmp marker
    if [[ -d "$STAGE1_CKPT" ]] && ! ls "${STAGE1_CKPT}/../"*.orbax-checkpoint-tmp-* 2>/dev/null | grep -q .; then
        CKPT_OK=true
    fi

    # norm_stats: file must exist and be non-empty
    if [[ -s "$NORM_STATS" ]]; then
        NORM_OK=true
    fi

    if $CKPT_OK && $NORM_OK; then
        log "Both prerequisites satisfied — launching Stage 2."
        break
    fi

    $CKPT_OK || log "  [wait] Stage 1 checkpoint not ready"
    $NORM_OK || log "  [wait] norm_stats not ready"
    sleep "$POLL_INTERVAL"
done

# ── Launch in tmux windows ────────────────────────────────────────────────────
SESSION="$(tmux display-message -p '#S' 2>/dev/null || echo 'main')"

log "Launching Variant A (GPU 0-3) in tmux window 'libero_A'..."
tmux new-window -t "$SESSION" -n "libero_A" \
    "cd $REPO_ROOT && CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_libero.sh A libero_r_A_${EXP_SUFFIX}; exec bash"

sleep 5  # brief pause to avoid JAX device contention at init

log "Launching Variant C (GPU 4-7) in tmux window 'libero_C'..."
tmux new-window -t "$SESSION" -n "libero_C" \
    "cd $REPO_ROOT && CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train_libero.sh C libero_r_C_${EXP_SUFFIX}; exec bash"


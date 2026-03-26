#!/usr/bin/env bash
# Wait for C eval to finish, kill C server, then run A eval (step 24999, clean mode).
#
# Usage:
#   bash scripts/watch_C_then_eval_A.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

C_EVAL_LOG="$REPO_ROOT/eval-logs/libero_C_libero_r_C_v1_step24999/eval.log"
C_SERVER_PID=3344315

echo "=== Watching C eval log for completion ==="
echo "  log : $C_EVAL_LOG"
echo "  C server PID: $C_SERVER_PID"
echo ""

# Wait until "=== Done." appears in C's eval log
while true; do
    if grep -q "=== Done\." "$C_EVAL_LOG" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] C eval finished."
        break
    fi
    sleep 10
done

# Show C's final results
echo ""
echo "=== C step24999 Results ==="
grep -E "Total success rate|total success|clean success rate|task success rate" "$C_EVAL_LOG" | tail -10
echo ""

# Kill C policy server
echo "Killing C policy server (PID $C_SERVER_PID) ..."
kill "$C_SERVER_PID" 2>/dev/null || true
wait "$C_SERVER_PID" 2>/dev/null || true
sleep 3

# Run A eval
echo ""
echo "=== Starting A step24999 eval ==="
bash "$SCRIPT_DIR/eval_libero.sh" A libero_r_A_v1 24999

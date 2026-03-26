#!/usr/bin/env bash
# Launch SS-v2 physical understanding pre-training.
#
# Usage:
#   bash scripts/train_ssv2.sh [EXP_NAME] [-- extra train.py args]
#
# Examples:
#   bash scripts/train_ssv2.sh ssv2_concat_v1
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/train_ssv2.sh ssv2_start_only -- --data.input_mode start
#   bash scripts/train_ssv2.sh ssv2_concat_v1 -- --overwrite
#
# Environment variables:
#   WANDB_API_KEY   : if set, used to authenticate with W&B automatically
#   WANDB_ENTITY    : W&B entity / username (optional, falls back to wandb default)
#   CUDA_VISIBLE_DEVICES : restrict which GPUs to use

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXP_NAME="${1:-ssv2_concat_v1}"
shift || true

# Remaining args after "--" are passed through to train.py
EXTRA_ARGS=()
pass_through=false
for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        pass_through=true
        continue
    fi
    $pass_through && EXTRA_ARGS+=("$arg")
done

# ── Paths ────────────────────────────────────────────────────────────────────
LABELS_PATH="$REPO_ROOT/ss-v2/labels/filtered_train.json"
FRAMES_DIR="$REPO_ROOT/../datasets/20bn-something-something-v2-frames"

if [[ ! -f "$LABELS_PATH" ]]; then
    echo "ERROR: labels file not found: $LABELS_PATH"
    exit 1
fi
if [[ ! -d "$FRAMES_DIR" ]]; then
    echo "ERROR: frames dir not found: $FRAMES_DIR"
    exit 1
fi

# ── W&B setup ────────────────────────────────────────────────────────────────
# Load .env if present (supports WANDB_API_KEY=... lines)
if [[ -f "$REPO_ROOT/.env" ]]; then
    echo "Loading .env ..."
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

PYTHON="$REPO_ROOT/.venv/bin/python"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "Logging in to W&B ..."
    "$PYTHON" -m wandb login "$WANDB_API_KEY" --relogin
else
    # Check if already logged in
    CURRENT_KEY=$("$PYTHON" -c "import wandb; print(wandb.api.api_key or '')" 2>/dev/null || true)
    if [[ -z "$CURRENT_KEY" ]]; then
        echo ""
        echo "W&B is not authenticated."
        echo "Set WANDB_API_KEY=<your-key> or put it in $REPO_ROOT/.env, then re-run."
        echo "Alternatively: .venv/bin/wandb login"
        echo ""
        echo "Continuing with wandb disabled for now (set wandb_enabled=False in config or login first)."
        EXTRA_ARGS+=("--wandb_enabled" "False")
    fi
fi

# ── Launch ───────────────────────────────────────────────────────────────────
echo ""
echo "=== SS-v2 Pre-training ==="
echo "  exp_name   : $EXP_NAME"
echo "  labels     : $LABELS_PATH"
echo "  frames_dir : $FRAMES_DIR"
echo "  project    : vla-hard-negative"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "  extra args : ${EXTRA_ARGS[*]}"
echo ""

cd "$REPO_ROOT"

uv run python scripts/train.py \
    pi0_fast_ssv2 \
    --exp_name "$EXP_NAME" \
    --data.labels_path "$LABELS_PATH" \
    --data.frames_dir "$FRAMES_DIR" \
    "${EXTRA_ARGS[@]}"

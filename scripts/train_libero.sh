#!/usr/bin/env bash
# Launch Stage 2 libero-10-r fine-tuning (Variant A / C comparison).
#
# Usage:
#   bash scripts/train_libero.sh [VARIANT] [EXP_NAME] [-- extra train.py args]
#
# VARIANT is the config suffix:
#   A — baseline:        pi0-FAST base → libero-10-r
#   B — CoT baseline:    pi0-FAST base → libero-10-r with chain-of-thought
#   C — Stage1:          Stage1-A2L → libero-10-r
#   D — Stage1 + CoT:    Stage1-A2L → libero-10-r with chain-of-thought
#
# Examples:
#   bash scripts/train_libero.sh A libero_r_A_v1
#   bash scripts/train_libero.sh B libero_r_B_v1
#   bash scripts/train_libero.sh C libero_r_C_v1
#   bash scripts/train_libero.sh D libero_r_D_v1
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_libero.sh B libero_r_B_v1
#
# Environment variables:
#   WANDB_API_KEY        : if set, used to authenticate with W&B automatically
#   CUDA_VISIBLE_DEVICES : restrict which GPUs to use

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VARIANT="${1:-A}"
EXP_NAME="${2:-libero_r_${VARIANT}_v1}"
shift 2 || true

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

CONFIG_NAME="pi0_fast_libero_r_${VARIANT}"

# ── Validate ─────────────────────────────────────────────────────────────────
if [[ "$VARIANT" != "A" && "$VARIANT" != "B" && "$VARIANT" != "C" && "$VARIANT" != "D" ]]; then
    echo "ERROR: VARIANT must be A, B, C, or D (got: $VARIANT)"
    exit 1
fi

DATA_DIR="$REPO_ROOT/../datasets/libero-10-r"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: libero-10-r dataset not found: $DATA_DIR"
    exit 1
fi

# ── W&B setup ────────────────────────────────────────────────────────────────
if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env"
    set +a
fi

PYTHON="$REPO_ROOT/.venv/bin/python"

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    "$PYTHON" -m wandb login "$WANDB_API_KEY" --relogin
else
    CURRENT_KEY=$("$PYTHON" -c "import wandb; print(wandb.api.api_key or '')" 2>/dev/null || true)
    if [[ -z "$CURRENT_KEY" ]]; then
        echo "W&B not authenticated — continuing with wandb disabled."
        EXTRA_ARGS+=("--wandb_enabled" "False")
    fi
fi

# ── Launch ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Stage 2: libero-10-r fine-tuning ==="
echo "  variant    : $VARIANT"
echo "  config     : $CONFIG_NAME"
echo "  exp_name   : $EXP_NAME"
echo "  data_dir   : $DATA_DIR"
echo "  project    : vla-hard-negative"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "  extra args : ${EXTRA_ARGS[*]}"
echo ""

cd "$REPO_ROOT"

uv run python scripts/train.py \
    "$CONFIG_NAME" \
    --exp_name "$EXP_NAME" \
    "${EXTRA_ARGS[@]}"

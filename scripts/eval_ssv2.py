"""SSv2 A2L evaluation: generation quality for physical understanding pre-training.

Metrics:
  exact_match  : % of samples where generated text == correct label (case-insensitive, stripped)
  rouge_l      : average ROUGE-L F1 between generated text and correct label
  diversity    : unique outputs / total samples (collapse detector)
  mean_len     : average output length in tokens (generic output detector)

Smoke test mode (--smoke):
  Runs 100 samples only — use before long training runs to catch trivial solutions early.

Usage:
    uv run python scripts/eval_ssv2.py \\
        --checkpoint checkpoints/pi0_fast_ssv2/ssv2_a2l_v1/1000 \\
        --labels ss-v2/labels/filtered_validation.json \\
        --frames ../datasets/20bn-something-something-v2-frames \\
        --num_samples 500

    # Smoke test (fast, run at step ~200-500):
    uv run python scripts/eval_ssv2.py \\
        --checkpoint checkpoints/pi0_fast_ssv2/ssv2_a2l_v1/500 \\
        --labels ss-v2/labels/filtered_validation.json \\
        --frames ../datasets/20bn-something-something-v2-frames \\
        --smoke
"""

import argparse
import json
import random
import re
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.training.config as _config

_IMAGE_H = 224
_IMAGE_W = 224
PALIGEMMA_EOS_TOKEN = 1


# ── ROUGE-L ────────────────────────────────────────────────────────────────────

def _lcs_length(a: list, b: list) -> int:
    """Longest common subsequence length."""
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(m):
        prev = 0
        for j in range(n):
            temp = dp[j + 1]
            dp[j + 1] = prev + 1 if a[i] == b[j] else max(dp[j + 1], dp[j])
            prev = temp
    return dp[n]


def rouge_l(pred: str, ref: str) -> float:
    p_tokens = pred.strip().lower().split()
    r_tokens = ref.strip().lower().split()
    if not p_tokens or not r_tokens:
        return 0.0
    lcs = _lcs_length(p_tokens, r_tokens)
    precision = lcs / len(p_tokens)
    recall = lcs / len(r_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Tokenizer helpers ──────────────────────────────────────────────────────────

def build_prefix_tokens(pg_tok, max_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokenize the A2L task prefix. ar_mask = 0 throughout (no autoregression in prefix)."""
    prefix_text = (
        "You are given three frames from a video in order: "
        "image 1 (start of action), image 2 (middle of action), image 3 (end of action).\n"
        "Describe the action shown across the three frames.\n"
    )
    tokens = pg_tok.encode(prefix_text, add_bos=True)

    n = len(tokens)
    if n < max_len:
        pad = max_len - n
        tokens = tokens + [0] * pad
        mask = [True] * n + [False] * pad
        ar = [0] * max_len
    else:
        tokens = tokens[:max_len]
        mask = [True] * max_len
        ar = [0] * max_len

    return (
        np.asarray(tokens, dtype=np.int32),
        np.asarray(mask, dtype=bool),
        np.asarray(ar, dtype=np.int32),
    )


def decode_output(pg_tok, token_ids: np.ndarray) -> str:
    token_ids = token_ids.astype(np.int32)
    eos_pos = np.where(token_ids == PALIGEMMA_EOS_TOKEN)[0]
    if len(eos_pos) > 0:
        token_ids = token_ids[: eos_pos[0]]
    text = pg_tok.decode(token_ids.tolist())
    text = re.sub(r"(?i)^answer:\s*", "", text).strip()
    return text


# ── Frame loading ──────────────────────────────────────────────────────────────

def load_frame(frames_dir: Path, video_id: str, which: str) -> np.ndarray:
    path = frames_dir / f"{video_id}_{which}.jpg"
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != (_IMAGE_H, _IMAGE_W):
        img = cv2.resize(img, (_IMAGE_W, _IMAGE_H), interpolation=cv2.INTER_LINEAR)
    return img


def run_batch(model, pg_tok, obs, max_decoding_steps, temperature, rng):
    rng, step_rng = jax.random.split(rng)
    output_tokens = model.sample_actions(
        step_rng, obs,
        max_decoding_steps=max_decoding_steps,
        temperature=temperature,
    )
    output_tokens = np.asarray(output_tokens)
    preds = [decode_output(pg_tok, output_tokens[i]) for i in range(output_tokens.shape[0])]
    return preds, rng


def build_obs(frames_dir, entries, model_config, pg_tok, max_token_len):
    B = len(entries)
    base0 = np.stack([load_frame(frames_dir, e["id"], "start") for e in entries])
    base1 = np.stack([load_frame(frames_dir, e["id"], "middle") for e in entries])
    base2 = np.stack([load_frame(frames_dir, e["id"], "end") for e in entries])

    tok_batch = [build_prefix_tokens(pg_tok, max_token_len) for _ in entries]
    tokens = np.stack([t[0] for t in tok_batch])
    token_mask = np.stack([t[1] for t in tok_batch])
    ar_mask = np.stack([t[2] for t in tok_batch])

    return _model.Observation.from_dict({
        "image": {
            "base_0_rgb": base0,
            "base_1_rgb": base1,
            "base_2_rgb": base2,
        },
        "image_mask": {
            "base_0_rgb": np.ones(B, dtype=bool),
            "base_1_rgb": np.ones(B, dtype=bool),
            "base_2_rgb": np.ones(B, dtype=bool),
        },
        "state": np.zeros((B, model_config.action_dim), dtype=np.float32),
        "tokenized_prompt": tokens,
        "tokenized_prompt_mask": token_mask,
        "token_ar_mask": ar_mask,
    })


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels", default="ss-v2/labels/filtered_validation.json")
    parser.add_argument("--frames", required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_decoding_steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--config", default="pi0_fast_ssv2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test mode: 100 samples only, print collapse warnings")
    args = parser.parse_args()

    if args.smoke:
        args.num_samples = 100
        print("[smoke test] Running 100 samples — checking for trivial solutions")

    random.seed(args.seed)
    frames_dir = Path(args.frames)
    params_dir = Path(args.checkpoint) / "params"
    assert params_dir.exists(), f"params dir not found: {params_dir}"

    train_config = _config.get_config(args.config)
    model_config: pi0_fast.Pi0FASTConfig = train_config.model
    max_token_len = model_config.max_token_len

    print(f"Loading params from {params_dir} ...")
    params = _model.restore_params(params_dir, dtype=jnp.bfloat16)
    model = model_config.load(params)
    model.eval()

    fast_tok = _tokenizer.FASTTokenizer(max_token_len)
    pg_tok = fast_tok._paligemma_tokenizer

    with open(args.labels) as f:
        all_entries = json.load(f)

    valid_entries = []
    for e in all_entries:
        vid = e["id"]
        if all((frames_dir / f"{vid}_{w}.jpg").exists() for w in ("start", "middle", "end")):
            valid_entries.append(e)
        if len(valid_entries) >= args.num_samples:
            break

    print(f"Evaluating {len(valid_entries)} samples ...")

    rng = jax.random.key(args.seed)
    all_preds = []
    all_refs = []

    for batch_start in range(0, len(valid_entries), args.batch_size):
        batch = valid_entries[batch_start: batch_start + args.batch_size]
        obs = build_obs(frames_dir, batch, model_config, pg_tok, max_token_len)
        preds, rng = run_batch(model, pg_tok, obs, args.max_decoding_steps, args.temperature, rng)

        for i, e in enumerate(batch):
            ref = e["label"].lower().strip().replace("_", " ")
            pred = preds[i].strip().lower()
            all_preds.append(pred)
            all_refs.append(ref)

            em = pred == ref
            rl = rouge_l(pred, ref)
            total = len(all_preds)
            print(
                f"[{total:4d}]  EM={'✓' if em else '✗'}  RL={rl:.2f}"
                f"  pred: {pred[:60]}"
            )

    total = len(all_preds)
    exact_matches = sum(p == r for p, r in zip(all_preds, all_refs))
    avg_rouge_l = sum(rouge_l(p, r) for p, r in zip(all_preds, all_refs)) / total
    unique_outputs = len(set(all_preds))
    diversity = unique_outputs / total
    mean_len = sum(len(p.split()) for p in all_preds) / total

    print(f"\n{'='*60}")
    print(f"Samples      : {total}")
    print(f"Exact match  : {exact_matches}/{total} = {exact_matches/total:.3f}")
    print(f"ROUGE-L      : {avg_rouge_l:.3f}")
    print(f"Diversity    : {unique_outputs}/{total} = {diversity:.3f}  (unique outputs / total)")
    print(f"Mean length  : {mean_len:.1f} words")
    print(f"{'='*60}")

    if args.smoke:
        print("\n[smoke test] Collapse checks:")
        if diversity < 0.1:
            print(f"  ⚠ LOW DIVERSITY ({diversity:.3f}) — model may be collapsing to fixed outputs")
        else:
            print(f"  ✓ diversity ok ({diversity:.3f})")
        if mean_len < 2.0:
            print(f"  ⚠ SHORT OUTPUTS ({mean_len:.1f} words) — model may be outputting generic/empty text")
        else:
            print(f"  ✓ output length ok ({mean_len:.1f} words)")
        if avg_rouge_l < 0.05:
            print(f"  ⚠ VERY LOW ROUGE-L ({avg_rouge_l:.3f}) — training signal may not be working")
        else:
            print(f"  ✓ rouge-l ok ({avg_rouge_l:.3f})")


if __name__ == "__main__":
    main()

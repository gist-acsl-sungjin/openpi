# Research Note: SS-v2 Pre-training Pipeline for VLA Physical Understanding

**Date:** 2026-03-18 (updated)
**Author:** Sungjin
**Status:** Design revised — implementation in progress

---

## 1. Motivation

pi0-FAST (base model) is trained on successful robot trajectories only. It lacks the ability to discriminate between correct and incorrect instructions for a given scene. To address this, we propose a **2-stage training pipeline**:

1. **Stage 1 (this note):** Pre-train on Something-Something-v2 (SS-v2) for physical understanding — teach the model to verify whether an instruction matches what is happening in a video sequence.
2. **Stage 2:** Fine-tune on NVIDIA/libero-r-datasets for robot action alignment with hard negative text swapping.

---

## 2. Training Signal Design (Revised 2026-03-18)

### Previous design (deprecated)
- Input: image(s) + (possibly wrong) instruction → Output: correct instruction text
- **Problem:** Model learned to copy the input label (trivial solution). No real scene understanding.

### New design: Yes/No Verification + Description

50% positive / 50% negative split per batch.

| Case | Input | Output |
|------|-------|--------|
| Positive (50%) | correct label | `"Yes"` |
| Negative (50%) | wrong label | `"No. The actual action is: {correct_label}"` |

**Prompt template:**
```
Task: Watch the sequence of frames carefully.
Does the following instruction accurately describe the action shown?
Instruction: "{label}"
Answer 'Yes' if correct, or 'No. The actual action is: [description]' if incorrect.
```

**Why this works:**
- Model cannot copy input for "Yes" case — must verify against visual content
- "No + description" forces understanding of what IS happening, not just what was asked
- Consistent with PaliGemma's VQA-style pretraining

---

## 3. Dataset: Something-Something-v2

### Category Selection

**Type A — Re-classified pairs (8 pairs):**

On closer inspection, only 2 of the 8 pairs are true success/failure pairs. The remaining 6 are two different valid outcomes of similar actions.

| pair_id | Positive | Negative | Classification |
|---------|----------|----------|----------------|
| put_into | Putting [X] into [Y] | **Failing** to put [X] into [Y] | ✅ True failure pair |
| attach | Attaching [X] to [Y] | **Trying but failing** to attach [X] to [Y] | ✅ True failure pair |
| poke_stack | Stack collapses | Stack does NOT collapse | ↔ Different outcomes |
| pull_apart | Separates | Nothing happens | ↔ Different outcomes |
| upright | Stands upright | Falls on its side | ↔ Different outcomes |
| tilt | Falls off | Doesn't fall | ↔ Different outcomes |
| slant_slide | Slides down | Stays | ↔ Different outcomes |
| plug | Stays in | Pulled right out | ↔ Different outcomes |

**Negative generation strategy by type:**

| Source | Negative strategy | Hardness |
|--------|-------------------|----------|
| Type A true failure (`put_into`, `attach`) | Explicit failure label | Hard — semantically close |
| Type A different outcomes (6 pairs) | Cross-pair label swap (show video A, use label B from same pair) | Medium-Hard — same objects/scene |
| Type B curated singles | In-batch random label swap | Easy-Medium |

**Why cross-pair swap is better than in-batch for Type A (6 pairs):**
Same objects, same scene, different physical outcome → visually challenging to distinguish.

**Type B — Curated Singles (~50 categories):**
Robot-relevant manipulation categories with spatial/relational language.
Negatives: in-batch random label swap at training time.
Excluded: "Pretending to...", camera movements, social/human actions.

### Statistics (filtered_train.json)
- Total kept: **68,915** entries (40.8% of original 168,913)
- Pair entries: **10,420** (both positive and negative sides)
- Single entries: **58,495**

### Data Fields
```json
{
  "id": "78687",
  "label": "poking a stack of tiles without the stack collapsing",
  "template": "Poking a stack of [something] without the stack collapsing",
  "placeholders": ["tiles"],
  "is_negative": true,
  "pair_id": "poke_stack",
  "positive_label": "Poking a stack of tiles so the stack collapses"
}
```

Note: `is_negative` field in the JSON now means "this entry is the non-default-outcome side of a pair" — the true positive/negative semantic is determined by the new Yes/No training format at training time.

---

## 4. Image Input Design (Revised 2026-03-18)

### Camera Slot Mapping: 2 frames → 3 frames

**Previous:** start frame + end frame (no temporal information between)
**Problem:** Model sees before/after state but not HOW the action happened.
**New:** start + middle + end frame using all 3 camera slots.

| Slot | SS-v2 Pre-training | libero-r Fine-tuning |
|------|-------------------|----------------------|
| `base_0_rgb` | Start frame (224×224) | 3rd person view |
| `base_1_rgb` | Middle frame (224×224) | masked out |
| `base_2_rgb` | End frame (224×224) | masked out |

Note: End frame uses `base_2_rgb` (not `wrist_0_rgb`) so that `model.py`'s augmentation does not
treat it differently — "wrist" in the key name skips crop/rotate, which would create asymmetric
augmentation across the three frames and bias the model toward the end frame.

**Frame sampling:** Uniform 3-split from video duration. SS-v2 videos ~2–6s at 12fps = 24–72 frames → frame indices at 0%, 50%, 100% of video length.

**Why `wrist_0_rgb` slot for end frame is OK at fine-tuning:**
This slot is used for wrist camera in libero-r. During Stage 1 pretraining it holds the end frame. The slots serve different roles per stage, both handled by `image_masks`.

### Ablation: `input_mode` parameter (still valid)
- `"3frame"` (new default): start → `base_0_rgb`, middle → `base_1_rgb`, end → `wrist_0_rgb`
- `"2frame"`: start → `base_0_rgb`, end → `base_1_rgb` (previous default, now baseline)
- `"start"`: only start frame
- `"end"`: only end frame

---

## 5. Tokenization: TokenizeSSv2Inputs (Revised)

**New token layout:**
```
Prefix  (loss_mask=False): [BOS] "Task: Watch the sequence of frames carefully.\n
                                  Does the following instruction accurately describe the action shown?\n
                                  Instruction: \"{label}\"\n"
Postfix (loss_mask=True):  "Yes" [EOS]                          ← positive
                           "No. The actual action is: {correct_label}" [EOS]  ← negative
```

Uses PaliGemma tokenizer directly. No FAST action tokens in Stage 1.

---

## 6. LoRA Configuration

### Stage 1 (SS-v2)
- **Use LoRA** — physical understanding is a soft prior; must not degrade action generation capability
- Target: PaliGemma language backbone attention layers only
- Action head: **frozen**
- Rank: **16**, alpha: 32
- Rationale: VQA-style task, low complexity → rank 16 sufficient

### Stage 2 (libero-r)
- Language backbone: **LoRA** (rank 32, alpha 64) — preserves Stage 1 prior while adapting
- Action head (FAST tokenizer): **full fine-tune** — action tokens are new, must be learned from scratch per domain
- Rationale: action generation requires full capacity in the action head

---

## 7. Training Schedule

### Stage 1 (SS-v2)
- Steps: **20k** (baseline), extend to 30k if not converged
- Batch size: 64 (can increase to 128 with LoRA memory savings)
- LR schedule: warmup 500 → peak 1e-4 → cosine decay to 1e-5 over 20k steps
- Eval: `eval_ssv2.py` exact-match accuracy at 5k, 10k, 20k checkpoints
- Dataset: 68,915 / 64 ≈ 1,077 steps/epoch → 20k steps ≈ 18.6 epochs

### Stage 2 (libero-r)
- Steps: **50k–80k** (start 50k, extend based on success rate curve)
- Batch size: 32–64
- LR schedule: warmup 1k → peak **5e-5** → cosine decay to 5e-6 (lower LR than Stage 1 to protect backbone)
- Eval: LIBERO task success rate

---

## 8. File Structure

```
openpi/
├── ss-v2/
│   ├── labels/
│   │   ├── train.json               # original SS-v2 train labels
│   │   ├── filtered_train.json      # filtered + annotated (68,915 entries)
│   │   └── ...
│   ├── filter_ssv2.py               # filtering + positive_label generation
│   ├── extract_frames.py            # extracts start/middle/end frames from webm videos
│   └── RESEARCH_NOTE.md             # this file

src/openpi/
├── models/
│   └── model.py                     # added: Observation.is_negative field
├── training/
│   ├── ssv2_dataset.py              # SSv2Dataset — to be updated for 3-frame + Yes/No format
│   ├── data_loader.py               # wired: SSv2Dataset for repo_id="ssv2"
│   └── config.py                    # SSv2DataConfig + pi0_fast_ssv2 TrainConfig + LoRA config
└── transforms.py                    # TokenizeSSv2Inputs — to be updated for new prompt format

datasets/
├── 20bn-something-something-v2-frames/   # 137,830 jpgs (start+end only — needs re-extraction for middle frames)
└── 20bn-something-something-v2.tar.gz    # archived original videos (5.6G)
```

---

## 9. Observation.is_negative Field

Added to `model.py`:
```python
is_negative: at.Bool[ArrayT, "*b"] | None = None
```

`from_dict` materializes a default `np.zeros(batch_shape, bool)` using `data["state"].shape[:-1]` when key is absent. This ensures JAX tracing always sees an array (no None branching in jit).

---

## 10. Implementation TODOs (updated 2026-03-19)

### Stage 1 (완료)
- [x] `extract_frames.py`: middle frame 추가 (uniform 3-split)
- [x] `SSv2Dataset`: A2L 방식으로 재설계 (neg 로직 제거, `target_prompt = label`)
- [x] `TokenizeSSv2Inputs`: A2L prefix ("Describe the action shown across the three frames.")
- [x] `config.py`: LoRA rank 16, `neg_prob` 제거
- [x] `eval_ssv2.py`: ROUGE-L + exact match + diversity + `--smoke` 플래그
- [x] **5k step 훈련 완료** (`ssv2_a2l_v1/4999`) — smoke test 통과 (RL=0.455, div=0.99)

### Stage 2 (훈련 완료 — Eval 진행 중)
- [x] Variant A config (`pi0_fast_libero_r_A`): base → libero-10-r, LoRA rank 16, 25k steps
- [x] Variant C config (`pi0_fast_libero_r_C`): Stage1-A2L → libero-10-r, loads `ssv2_a2l_v1/9999/params`
- [x] `train_libero.sh` launch script (GPU 분리 지원: A=0~3, C=4~7)
- [x] `wait_and_launch.sh` watcher 스크립트 (Stage 1 완료 감지 → 자동 Stage 2 시작)
- [x] libero-10-r norm_stats 완료 (`assets/libero-10-r/norm_stats.json`)
- [x] `eval_libero.sh` eval 스크립트 완료 (serve_policy + libero main.py client)
- [x] **Variant A/C 25k step 훈련 완료**
- [x] **LIBERO-10 Eval 실행 중** (A: port 8000, C: port 8001)
- [ ] Eval 결과 확인 및 A vs C 비교 분석
- [ ] libero-r CoT-based training (Variant B/D)
- [ ] A2L 포함 학습 variant (Variant E)
- [ ] L2C verifier + verifier steering (Variant F)

### Ablations (우선순위 순)
- [ ] Stage 1 smoke test: step 1000 에서 ROUGE-L / diversity 확인
- [ ] Stage 2 [A]: base → 일반 학습 → robot eval (베이스라인)
- [ ] Stage 2 [C]: Stage1-A2L → 일반 학습 → robot eval (Stage 1 효과 확인)
- [ ] Stage 2 [B]: base → CoT 학습 → robot eval
- [ ] Stage 2 [D]: Stage1-A2L → CoT 학습 → robot eval
- [ ] Stage 2 [E]: Stage1-A2L → CoT+A2L 학습 → robot eval
- [ ] Stage 2 [F]: Stage1-A2L → CoT+A2L 학습 → verifier steering

---

## 11. 실험 매트릭스 (updated 2026-03-19)

| ID | Stage 1 | Stage 2 학습 | Inference | 목적 |
|----|---------|-------------|-----------|------|
| A | base | 일반 | plain | 베이스라인 |
| B | base | CoT | plain | CoT 효과 |
| C | Stage1-A2L | 일반 | plain | **Stage 1 효과** |
| D | Stage1-A2L | CoT | plain | Stage1 + CoT |
| E | Stage1-A2L | CoT+A2L | plain | A2L 학습 효과 |
| F | Stage1-A2L | CoT+A2L | verifier steering | **최종 목표** |

**핵심 비교:**
- A vs C: Stage 1 pretraining이 action 학습에 도움이 되는가
- A vs B: CoT가 도움이 되는가
- D vs E: A2L을 학습에 포함하는 것이 도움이 되는가
- E vs F: Verifier steering이 성능을 올리는가

**현재 상태 (2026-03-22):**
- Stage 1-A2L: **10k 완료** (`ssv2_a2l_v1/9999`)
- Stage 2: **Variant A/C 훈련 완료** (25k steps each)
  - Variant A checkpoint: `checkpoints/pi0_fast_libero_r_A/libero_r_A_v1/24999`
  - Variant C checkpoint: `checkpoints/pi0_fast_libero_r_C/libero_r_C_v1/24999`
- **LIBERO-10 Eval 실행 중**
  - Variant A: port 8000, `eval-logs/libero_A_libero_r_A_v1_step24999/`
  - Variant C: port 8001, `eval-logs/libero_C_libero_r_C_v1_step24999/`
  - MUJOCO_GL=osmesa (EGL PLATFORM_DEVICE 미지원 환경 → osmesa로 해결)
  - 20 trials/task, libero_10 suite (10 tasks)

---

## 12. Stage 1 Eval Results

### 5k step smoke test (`ssv2_a2l_v1/4999`, 2026-03-19)

| 지표 | 값 |
|------|-----|
| Exact Match | 5% (5/100) |
| ROUGE-L | **0.455** |
| Diversity | 0.990 (99/100) |
| Mean length | 6.2 words |

**Collapse 체크:** diversity ✓ / output length ✓ / rouge-l ✓ — 모두 통과

**정성적 평가:**
- Action verb 학습 완료 (moving, pushing, pulling, putting, pouring, tilting, uncovering, plugging 등 동작과 대응)
- Directional language 포착 ("from left to right", "away from each other") — 3-frame이 motion dynamics를 전달하고 있음
- SS-v2 템플릿 구조 `verb + object + preposition + location` 학습됨
- Object-level visual grounding은 아직 noisy (RL=0 케이스 대부분 object 오인식)
- 복잡한 라벨에서도 RL=0.94~0.97 달성 — 기본 구조는 수렴

**결론:** 5k step에서 action understanding의 기본 구조는 잡혔고, collapse 없음. 20k까지 진행 시 object grounding 개선 예상.

---

## 13. Future Direction: Grounding Problem (Option 2)

> **현재 scope (Option 1) 외의 방향 — 추후 연구 후보**

**문제 정의:**
모델이 올바른 내부 reasoning을 하더라도 (예: "1+1=2"), 그 reasoning을 실제 physical action으로 연결하지 못하는 grounding 문제.
(예: 2가 쓰인 카드를 집어야 할 때 집지 못함)

**Option 1의 한계:**
- External verifier (Variant F)는 reasoning-action 불일치를 **사후 필터링**할 뿐
- 모델 내부의 reasoning → action 연결 자체를 훈련하지 않음

**Option 2 방향:**
Stage 2에서 actalign-style internal reasoning-action alignment 도입:
- 모델이 reasoning token을 생성하고, 그것이 직접 action generation을 조건화
- `language → motor action` 방향의 grounding을 명시적으로 학습
- pi0_fuse 아키텍처 필요 → Stage 1 isolated effect 측정 불가 (trade-off)

**고려 시점:** Option 1 (A vs C 비교)으로 Stage 1 효과가 입증된 후, grounding gap이 여전히 존재한다면 Option 2를 다음 단계로 검토.

---

## 13. Key Design Decisions

1. **Why SS-v2?** Robot datasets have few failure examples. SS-v2 has natural variation pairs with explicit labels covering pick/place/plug/tilt/pull physics.

2. **Why A2L instead of Yes/No + description?**
   - v1 (instruction reproduction): trivial copying
   - v2 (Yes/No + description): "always yes" collapse — "yes"(1 token) vs "no + description"(20 tokens) loss asymmetry로 3000 step에도 neg accuracy = 0%
   - v3 (A2L): 입력에 정답 없음 → trivial solution 불가. LACY의 A2L 파트와 정확히 대응.

3. **Why 3 frames instead of 2?** Start/end only shows before/after state, not the action itself. Middle frame captures motion dynamics.

4. **Why LoRA for Stage 1?** Stage 1 is a soft prior — must not degrade action generation capability.

5. **Why zeros for state during SS-v2?** SS-v2 has no robot state. Keeps slot consistent with downstream format.

6. **Trivial solution checklist for A2L:**
   - Generic outputs ("person doing something") → diversity metric
   - Template memorization → per-template accuracy
   - Short/empty outputs → mean output length
   → Smoke test at step 1000 covers all three.

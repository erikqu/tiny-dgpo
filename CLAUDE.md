# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
uv sync

# Run DGPO training
uv run python train_dgpo.py

# Run GRPO baseline
uv run python train.py

# Run unit tests
uv run python test_dgpo.py

# Install flash-attn (optional, for faster attention)
uv pip install flash-attn --no-build-isolation
```

## Architecture

This is a minimal implementation of DGPO (Distribution-Guided Policy Optimization) built on top of GRPO (Group Relative Policy Optimization).

### Core Algorithm Flow

1. **Rollout**: Policy generates G completions per prompt, each scored by a verifier
2. **GRPO Advantage**: `A_i = (r_i - mean(r)) / std(r)` — sequence-level, same for all tokens
3. **DGPO Token Scoring**: For each token, compute:
   - `d_t` = Hellinger distance between policy and reference (bounded [0,1])
   - `H_t` = normalized entropy of policy distribution
   - `s_t = d_t * H_t^κ` — gated score filtering hallucinations
4. **Credit Reallocation**: `w_t = T * softmax(s_t / τ)`, then `A_t = A_i * w_t`
5. **Loss**: PPO-clip with token-level advantages, no KL penalty

### Key Differences from GRPO

- DGPO uses Hellinger distance (bounded) instead of KL divergence (unbounded)
- Entropy gating filters "fake innovations" (high deviation + low entropy = hallucination)
- Token-level advantage redistribution amplifies pivotal reasoning steps
- No explicit KL penalty in loss — deviation becomes a guiding signal, not a penalty

### Module Responsibilities

- `dgpo.py`: Pure functions for Hellinger distance, entropy, score computation, weight calculation
- `loss.py`: `GRPOLoss` and `DGPOLoss` classes wrapping the PPO-clip objective
- `replay_buffer.py`: `Experience` dataclass and batching utilities; stores ref_logits for DGPO
- `train_dgpo.py`: Main training loop — rollout, log-prob collection, optimization

### Memory Considerations

DGPO stores full vocabulary logits for Hellinger computation. For large vocab models (Qwen ~150k):
- Only `ref_logits` are stored (frozen reference model)
- Current policy logits computed fresh during training
- Use smaller batch/group sizes if OOM: `train_batch_size=4`, `group_size=8`

## Paper Hyperparameters

From DGPO paper (arXiv:2605.03327):
- τ (tau) = 0.5: softmax temperature
- κ (kappa) = 1.0: entropy gating exponent  
- lr = 1e-6, weight_decay = 0.1, clip_eps = 0.2

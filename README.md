# Minimal GRPO / DGPO Implementation

Minimal implementations of:
- **GRPO** (Group Relative Policy Optimization) — DeepSeek's critic-free RL algorithm
- **DGPO** (Distribution-Guided Policy Optimization) — Token-level credit assignment via Hellinger distance + entropy gating

## Setup

```bash
uv sync
```

## Usage

### GRPO (baseline)

```bash
uv run python train.py
```

### DGPO (token-level reweighting)

```bash
uv run python train_dgpo.py
```

### Run tests

```bash
uv run python test_dgpo.py
```

## DGPO Algorithm

DGPO improves on GRPO by redistributing the sequence-level advantage to individual tokens based on:

1. **Hellinger distance** `d_{i,t}` — How much the policy deviates from the reference at each token (bounded [0,1], unlike KL)
2. **Entropy gating** `H_{i,t}^κ` — Filters out "fake innovations" (high deviation + low uncertainty = hallucination)
3. **Token weights** `w_{i,t} = T_i * softmax(d * H^κ / τ)` — Amplifies credit for exploratory steps, discounts routine syntax

The final token-level advantage `A_{i,t} = A_i * w_{i,t}` preserves total credit while focusing learning on pivotal tokens.

Key hyperparameters:
- `tau` (τ): Temperature for softmax reweighting (0.5-1.0 recommended)
- `kappa` (κ): Entropy gating exponent (1.0 recommended)

## Files

| File | Description |
|------|-------------|
| `dgpo.py` | Core DGPO functions (Hellinger, entropy, reweighting) |
| `loss.py` | GRPOLoss and DGPOLoss classes |
| `train.py` | GRPO training loop |
| `train_dgpo.py` | DGPO training loop |
| `replay_buffer.py` | Experience storage and batching |
| `test_dgpo.py` | Unit tests for DGPO math |

## References

- [DGPO Paper (arXiv:2605.03327)](https://arxiv.org/abs/2605.03327) — Distribution-Guided Policy Optimization for Fine-Grained Credit Assignment
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) — GRPO origin
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — Mathematical reasoning with GRPO
- [tiny-grpo (upstream)](https://github.com/open-thought/tiny-grpo) — Original minimal GRPO implementation

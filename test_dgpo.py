#!/usr/bin/env python3
"""Unit tests for DGPO core functions."""

import torch
from dgpo import (
    hellinger_distance,
    normalized_entropy,
    compute_dgpo_scores,
    compute_token_weights,
    compute_token_level_advantages,
)


def test_hellinger_distance():
    """Test Hellinger distance bounds and properties."""
    batch, seq, vocab = 2, 5, 100

    # Identical distributions -> distance = 0
    logits = torch.randn(batch, seq, vocab)
    d = hellinger_distance(logits, logits)
    assert d.shape == (batch, seq)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-5), f"Same dist should give 0, got {d}"

    # Very different distributions -> distance close to 1
    logits1 = torch.zeros(batch, seq, vocab)
    logits1[..., 0] = 100  # delta on token 0
    logits2 = torch.zeros(batch, seq, vocab)
    logits2[..., 1] = 100  # delta on token 1
    d = hellinger_distance(logits1, logits2)
    assert (d > 0.99).all(), f"Disjoint distributions should give ~1, got {d}"

    # Bounds check
    logits1 = torch.randn(batch, seq, vocab)
    logits2 = torch.randn(batch, seq, vocab)
    d = hellinger_distance(logits1, logits2)
    assert (d >= 0).all() and (d <= 1).all(), f"Hellinger should be in [0,1], got min={d.min()}, max={d.max()}"

    print("Hellinger distance tests passed!")


def test_normalized_entropy():
    """Test normalized entropy bounds and properties."""
    batch, seq, vocab = 2, 5, 100

    # Uniform distribution -> entropy = 1
    logits = torch.zeros(batch, seq, vocab)
    h = normalized_entropy(logits)
    assert h.shape == (batch, seq)
    assert torch.allclose(h, torch.ones_like(h), atol=1e-4), f"Uniform should give 1, got {h}"

    # Delta distribution -> entropy close to 0
    logits = torch.zeros(batch, seq, vocab)
    logits[..., 0] = 100
    h = normalized_entropy(logits)
    assert (h < 0.01).all(), f"Delta distribution should give ~0, got {h}"

    # Bounds check
    logits = torch.randn(batch, seq, vocab)
    h = normalized_entropy(logits)
    assert (h >= 0).all() and (h <= 1).all(), f"Entropy should be in [0,1], got min={h.min()}, max={h.max()}"

    print("Normalized entropy tests passed!")


def test_token_weights():
    """Test token weights sum to seq_len and are positive."""
    batch, seq = 2, 10
    scores = torch.rand(batch, seq)
    action_mask = torch.ones(batch, seq, dtype=torch.bool)

    # Mask out some positions
    action_mask[0, -2:] = False
    action_mask[1, -3:] = False

    weights = compute_token_weights(scores, action_mask, tau=0.5)

    # Check weights sum to seq_len for each sequence
    for i in range(batch):
        seq_len = action_mask[i].sum().item()
        w_sum = weights[i].sum().item()
        assert abs(w_sum - seq_len) < 1e-4, f"Weights should sum to {seq_len}, got {w_sum}"

    # Check weights are non-negative
    assert (weights >= 0).all(), "Weights should be non-negative"

    # Check masked positions have zero weight
    assert (weights[~action_mask] == 0).all(), "Masked positions should have zero weight"

    print("Token weights tests passed!")


def test_token_weights_all_masked_row():
    """Test token weights stay finite when a sequence has no valid actions."""
    scores = torch.rand(2, 10)
    action_mask = torch.ones(2, 10, dtype=torch.bool)
    action_mask[0, :] = False

    weights = compute_token_weights(scores, action_mask, tau=0.5)

    assert torch.isfinite(weights).all(), "Weights should stay finite"
    assert (weights[0] == 0).all(), "All-masked rows should have zero weights"
    assert abs(weights[1].sum().item() - action_mask[1].sum().item()) < 1e-4

    print("All-masked token weights tests passed!")


def test_token_level_advantages():
    """Test advantage redistribution preserves total credit."""
    batch, seq = 3, 8
    seq_advantages = torch.tensor([[2.0], [-1.0], [0.5]])  # [batch, 1]

    action_mask = torch.ones(batch, seq, dtype=torch.bool)
    scores = torch.rand(batch, seq)
    token_weights = compute_token_weights(scores, action_mask, tau=0.5)

    token_adv = compute_token_level_advantages(seq_advantages, token_weights)

    # Sum of token advantages should equal seq_len * seq_advantage
    for i in range(batch):
        total = token_adv[i].sum().item()
        expected = seq_advantages[i].item() * seq
        assert abs(total - expected) < 1e-4, f"Total advantage should be {expected}, got {total}"

    print("Token-level advantages tests passed!")


def test_dgpo_scores():
    """Test combined DGPO scores."""
    batch, seq, vocab = 2, 5, 100
    policy_logits = torch.randn(batch, seq, vocab)
    ref_logits = torch.randn(batch, seq, vocab)

    scores = compute_dgpo_scores(policy_logits, ref_logits, kappa=1.0)

    assert scores.shape == (batch, seq)
    assert (scores >= 0).all() and (scores <= 1).all(), "DGPO scores should be in [0,1]"

    # Same logits -> score = 0 (no deviation)
    scores = compute_dgpo_scores(policy_logits, policy_logits, kappa=1.0)
    assert torch.allclose(scores, torch.zeros_like(scores), atol=1e-5), "Same dist should give 0 score"

    print("DGPO scores tests passed!")


if __name__ == "__main__":
    test_hellinger_distance()
    test_normalized_entropy()
    test_token_weights()
    test_token_weights_all_masked_row()
    test_token_level_advantages()
    test_dgpo_scores()
    print("\nAll DGPO tests passed!")

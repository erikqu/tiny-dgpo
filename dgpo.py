"""
DGPO: Distribution-Guided Policy Optimization

Implements the core DGPO components from the paper:
- Hellinger distance between policy and reference distributions
- Normalized Shannon entropy for uncertainty gating
- Token-level advantage reweighting
"""

import torch
import torch.nn.functional as F


def hellinger_distance(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Hellinger distance between policy and reference distributions.

    d_{i,t} = 1 - sum_a sqrt(pi_theta(a|x, y<t) * pi_ref(a|x, y<t))

    The Hellinger distance is bounded in [0, 1], avoiding gradient explosion
    that occurs with unbounded KL divergence.

    Args:
        policy_logits: [batch, seq_len, vocab_size]
        ref_logits: [batch, seq_len, vocab_size]

    Returns:
        distances: [batch, seq_len] in [0, 1]
    """
    policy_probs = F.softmax(policy_logits.float(), dim=-1)
    ref_probs = F.softmax(ref_logits.float(), dim=-1)

    # Bhattacharyya coefficient: sum_a sqrt(p * q)
    bc = (policy_probs * ref_probs).sqrt().sum(dim=-1)

    # Hellinger distance: 1 - BC
    # Clamp to handle numerical issues
    return (1.0 - bc).clamp(min=0.0, max=1.0)


def normalized_entropy(
    logits: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute normalized Shannon entropy of the policy distribution.

    H_{i,t} = -sum_a p(a) log p(a) / log|V|

    Normalized to [0, 1] where:
    - 0 = deterministic (low uncertainty)
    - 1 = uniform (high uncertainty)

    Args:
        logits: [batch, seq_len, vocab_size]
        eps: small constant for numerical stability

    Returns:
        entropy: [batch, seq_len] in [0, 1]
    """
    vocab_size = logits.size(-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = F.softmax(logits.float(), dim=-1)

    # Shannon entropy: -sum p log p
    entropy = -(probs * log_probs).sum(dim=-1)

    # Normalize by max entropy (log |V|)
    max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32, device=logits.device))
    normalized = entropy / (max_entropy + eps)

    return normalized.clamp(min=0.0, max=1.0)


def compute_dgpo_scores(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Compute the joint DGPO score combining Hellinger distance and entropy gating.

    s_{i,t} = d_{i,t} * H_{i,t}^kappa

    This filters out "fake innovations" where deviation is high but entropy is low
    (confident hallucinations), while amplifying genuine exploratory steps where
    both deviation and uncertainty are high.

    Args:
        policy_logits: [batch, seq_len, vocab_size]
        ref_logits: [batch, seq_len, vocab_size]
        kappa: entropy gating scaling factor (paper recommends 1.0)

    Returns:
        scores: [batch, seq_len] in [0, 1]
    """
    d = hellinger_distance(policy_logits, ref_logits)
    h = normalized_entropy(policy_logits)

    # Entropy gating: scale deviation by uncertainty
    return d * (h ** kappa)


def compute_token_weights(
    scores: torch.Tensor,
    action_mask: torch.Tensor,
    tau: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Convert DGPO scores to token-level importance weights via temperature-scaled softmax.

    w_{i,t} = T_i * exp(s_{i,t} / tau) / sum_j exp(s_{i,j} / tau)

    The T_i scaling ensures that mean(w_{i,t}) = 1 across the sequence,
    preserving the overall gradient magnitude while redistributing credit.

    Args:
        scores: [batch, seq_len] DGPO scores s_{i,t}
        action_mask: [batch, seq_len] mask for valid tokens
        tau: temperature for softmax (lower = sharper, paper recommends 0.5-1.0)
        eps: numerical stability constant

    Returns:
        weights: [batch, seq_len] token importance weights, mean=1 per sequence
    """
    # Mask invalid positions with large negative value before softmax
    masked_scores = scores.clone()
    masked_scores[~action_mask] = float('-inf')

    # Temperature-scaled softmax
    softmax_weights = F.softmax(masked_scores / tau, dim=-1)

    # Scale by sequence length T_i to ensure unit mean
    seq_lengths = action_mask.sum(dim=-1, keepdim=True).float()
    weights = softmax_weights * seq_lengths

    # Zero out masked positions
    weights = weights * action_mask.float()

    return weights


def compute_token_level_advantages(
    sequence_advantages: torch.Tensor,
    token_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Redistribute sequence-level advantages to individual tokens.

    A_{i,t} = A_i * w_{i,t}

    This is the core DGPO insight: instead of broadcasting the same advantage
    to every token (GRPO), we amplify credit for pivotal exploratory steps
    and discount routine syntactic tokens.

    Args:
        sequence_advantages: [batch, 1] or [batch] group-relative advantages A_i
        token_weights: [batch, seq_len] importance weights w_{i,t}

    Returns:
        token_advantages: [batch, seq_len]
    """
    if sequence_advantages.dim() == 1:
        sequence_advantages = sequence_advantages.unsqueeze(-1)

    return sequence_advantages * token_weights

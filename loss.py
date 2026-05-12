from typing import Optional
import torch
import torch.nn as nn

from replay_buffer import Experience
from dgpo import compute_dgpo_scores, compute_token_weights, compute_token_level_advantages


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Monte-Carlo approximation of KL divergence, k3 estimator, see: http://joschu.net/blog/kl-approx.html
    """

    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """GRPO actor loss"""

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages

        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss, kl.mean()


class DGPOLoss(nn.Module):
    """
    DGPO actor loss with token-level advantage reweighting.

    Key differences from GRPO:
    1. No KL penalty - distribution deviation is used as a guiding signal, not a penalty
    2. Token-level advantages via Hellinger distance + entropy gating
    3. Still uses PPO-style clipping for stability
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        tau: float = 0.5,
        kappa: float = 1.0,
    ) -> None:
        """
        Args:
            clip_eps: PPO clip epsilon
            tau: temperature for softmax reweighting (0.5-1.0 recommended)
            kappa: entropy gating exponent (1.0 recommended)
        """
        super().__init__()
        self.clip_eps = clip_eps
        self.tau = tau
        self.kappa = kappa

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            log_probs: [batch, seq_len] current policy log probs for chosen tokens
            experience: Experience dataclass with action_log_probs, advantages, action_mask
            policy_logits: [batch, seq_len, vocab_size] current policy logits (full vocab)
            ref_logits: [batch, seq_len, vocab_size] reference model logits (full vocab)

        Returns:
            loss: scalar loss
            metrics: dict with dgpo-specific metrics for logging
        """
        old_log_probs = experience.action_log_probs
        action_mask = experience.action_mask
        sequence_advantages = experience.advantages  # [batch, 1] or [batch]

        # Compute DGPO token-level scores and weights
        scores = compute_dgpo_scores(
            policy_logits=policy_logits,
            ref_logits=ref_logits,
            kappa=self.kappa,
        )
        token_weights = compute_token_weights(
            scores=scores,
            action_mask=action_mask,
            tau=self.tau,
        )

        # Redistribute advantages to tokens
        token_advantages = compute_token_level_advantages(
            sequence_advantages=sequence_advantages,
            token_weights=token_weights,
        )

        # PPO-clip surrogate (no KL penalty)
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * token_advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * token_advantages
        loss = -torch.min(surr1, surr2)

        loss = masked_mean(loss, action_mask, dim=-1).mean()

        # Metrics for logging
        metrics = {
            "dgpo_score_mean": masked_mean(scores, action_mask).item(),
            "dgpo_weight_std": token_weights[action_mask].std().item() if action_mask.any() else 0.0,
        }

        return loss, metrics

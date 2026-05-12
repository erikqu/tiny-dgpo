from dataclasses import dataclass, fields
from typing import Optional, Self

import torch
import torch.nn.functional as F


def zero_pad_sequences(
    sequences: list[torch.Tensor], side: str = "left"
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


@dataclass
class Experience:
    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    log_probs_ref: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    action_mask: torch.Tensor
    kl: Optional[torch.Tensor] = None
    # DGPO: store full logits for Hellinger distance computation
    policy_logits: Optional[torch.Tensor] = None
    ref_logits: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> Self:
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)


def split_experience_batch(experience: Experience) -> list[Experience]:
    batch_size = experience.sequences.size(0)
    batch_data = [{} for _ in range(batch_size)]
    keys = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
        "policy_logits",
        "ref_logits",
    )
    for key in keys:
        value = getattr(experience, key)
        if value is None:
            vals = [None] * batch_size
        else:
            vals = torch.unbind(value)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_data[i][key] = v

    return [Experience(**data) for data in batch_data]


def zero_pad_sequences_3d(
    sequences: list[torch.Tensor], side: str = "left"
) -> torch.Tensor:
    """Pad 3D tensors (batch of [seq_len, vocab_size]) along seq_len dimension."""
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    vocab_size = sequences[0].size(1)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            pad_tensor = torch.zeros(pad_len, vocab_size, dtype=seq.dtype, device=seq.device)
            if side == "left":
                seq = torch.cat([pad_tensor, seq], dim=0)
            else:
                seq = torch.cat([seq, pad_tensor], dim=0)
        padded_sequences.append(seq)
    return torch.stack(padded_sequences, dim=0)


def join_experience_batch(items: list[Experience]) -> Experience:
    batch_data = {}
    keys_2d = (
        "sequences",
        "action_log_probs",
        "log_probs_ref",
        "returns",
        "advantages",
        "attention_mask",
        "action_mask",
    )
    keys_3d = (
        "policy_logits",
        "ref_logits",
    )
    for key in keys_2d:
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            data = zero_pad_sequences(vals, "left")
        else:
            data = None
        batch_data[key] = data
    for key in keys_3d:
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            data = zero_pad_sequences_3d(vals, "left")
        else:
            data = None
        batch_data[key] = data
    return Experience(**batch_data)


class ReplayBuffer:
    def __init__(self, limit: int = 0) -> None:
        self.limit = limit
        self.items: list[Experience] = []

    def append(self, experience: Experience) -> None:
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        return self.items[idx]

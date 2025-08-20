from typing import Optional

import torch


def get_last_attention_weights(model) -> Optional[torch.Tensor]:
    """Return the last stored attention weights for BiLSTMAttention models.

    Returns a tensor of shape (batch, seq) if available.
    """
    return getattr(model, "last_attn", None)

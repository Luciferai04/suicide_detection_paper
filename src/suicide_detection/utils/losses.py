from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FocalLossConfig:
    gamma: float = 2.0
    alpha_pos: Optional[float] = None  # if set for binary, weight positive class


class FocalLoss(nn.Module):
    """Focal Loss for binary or multi-class classification.

    - For multi-class, expects logits of shape (N, C) and targets of shape (N,).
    - For binary, expects logits of shape (N, 2) or (N,) with targets (N,).
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if isinstance(alpha, torch.Tensor) else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 1:
            # expand to 2-class
            logits = torch.stack([-logits, logits], dim=-1)
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.softmax(logits, dim=-1).gather(1, targets.view(-1, 1)).squeeze(1).clamp_min(1e-8)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, bidirectional: bool = True):
        super().__init__()
        self.scale = 1.0 / math.sqrt(hidden_dim * (2 if bidirectional else 1))
        self.query = nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, h: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (batch, seq, hidden*dirs)
        q = torch.tanh(self.query(h))  # (batch, seq, hidden)
        scores = self.v(q).squeeze(-1) * self.scale  # (batch, seq)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (batch, seq)
        context = torch.bmm(attn.unsqueeze(1), h).squeeze(1)  # (batch, hidden*dirs)
        return context, attn


@dataclass
class BiLSTMAttentionConfig:
    vocab_size: int = 50000
    embedding_dim: int = 200
    hidden_dim: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.3
    num_classes: int = 2


class BiLSTMAttention(nn.Module):
    def __init__(self, cfg: BiLSTMAttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
        )
        self.attn = Attention(cfg.hidden_dim, cfg.bidirectional)
        out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(out_dim, cfg.num_classes)
        self.last_attn: Optional[torch.Tensor] = None

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # input_ids: (batch, seq)
        x = self.embedding(input_ids)
        outputs, _ = self.lstm(x)
        context, attn_w = self.attn(outputs, mask=attention_mask)
        self.last_attn = attn_w.detach() if isinstance(attn_w, torch.Tensor) else None
        logits = self.fc(self.dropout(context))
        return logits

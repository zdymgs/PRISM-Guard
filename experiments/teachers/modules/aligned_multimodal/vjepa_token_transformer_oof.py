from __future__ import annotations
from _release_paths import WORK_ROOT, DATA_ROOT, PRETRAINED_ROOT, EXTERNAL_ROOT, MODULE_ROOT
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenHead(nn.Module):

    def __init__(self, input_dim: int=1024, hidden_dim: int=192, heads: int=6, layers: int=2, dropout: float=0.2, view_dropout: float=0.15, num_tokens: int=24) -> None:
        super().__init__()
        self.view_dropout = float(view_dropout)
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position = nn.Parameter(torch.zeros(1, int(num_tokens) + 1, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 3, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 40))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(values)
        padding = torch.zeros((len(values), tokens.shape[1] + 1), dtype=torch.bool, device=values.device)
        if self.training and self.view_dropout > 0:
            dropped = torch.rand((len(values), tokens.shape[1]), device=values.device) < self.view_dropout
            all_dropped = dropped.all(dim=1)
            if all_dropped.any():
                dropped[all_dropped, 0] = False
            tokens = tokens.masked_fill(dropped[..., None], 0.0)
            padding[:, 1:] = dropped
        cls = self.cls_token.expand(len(values), -1, -1)
        encoded = self.encoder(torch.cat((cls, tokens), dim=1) + self.position, src_key_padding_mask=padding)
        return encoded[:, 0]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(values))

def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=40).astype(np.float64)
    weights = np.sqrt(len(labels) / np.maximum(40.0 * counts, 1.0))
    weights /= weights.mean()
    return weights.astype(np.float32)

def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weighted_target = target * weights[None, :]
    loss = -(weighted_target * F.log_softmax(logits, dim=1)).sum(dim=1)
    normalizer = weighted_target.sum(dim=1).clamp_min(1e-06)
    return (loss / normalizer).mean()

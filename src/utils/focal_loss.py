"""Focal Loss for multi-class classification (Lin et al., 2017).

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0,
                 weight: torch.Tensor | None = None,
                 reduction: str = 'mean'):
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError(f'gamma must be non-negative, got {gamma}')
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction}")
        if weight is not None and weight.ndim != 1:
            raise ValueError(f'weight must be 1-D, got {weight.shape}')
        if weight is not None and (not torch.isfinite(weight).all() or torch.any(weight < 0)):
            raise ValueError('weight must contain finite non-negative values')
        self.gamma = gamma
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # input : (N, C) logits   target : (N,) class indices
        if input.ndim != 2:
            raise ValueError(f'input must have shape (N, C), got {input.shape}')
        if target.ndim != 1 or len(target) != len(input):
            raise ValueError(f'target must have shape ({len(input)},), got {target.shape}')
        if target.dtype != torch.long:
            raise ValueError(f'target dtype must be torch.long, got {target.dtype}')
        if target.numel() and (target.min() < 0 or target.max() >= input.shape[1]):
            raise ValueError('target contains a class index outside the input range')
        if self.weight is not None and len(self.weight) != input.shape[1]:
            raise ValueError(
                f'weight length {len(self.weight)} != number of classes {input.shape[1]}')
        log_p = F.log_softmax(input, dim=1)           # (N, C)
        pt    = torch.exp(log_p)                       # (N, C)

        log_pt = log_p.gather(1, target.view(-1, 1)).squeeze(1)   # (N,)
        pt_t   = pt.gather(1,    target.view(-1, 1)).squeeze(1)   # (N,)

        focal  = (1.0 - pt_t) ** self.gamma * (-log_pt)           # (N,)

        if self.weight is not None:
            alpha_t = self.weight[target]
            focal   = alpha_t * focal

        if self.reduction == 'mean':
            return focal.mean()
        if self.reduction == 'sum':
            return focal.sum()
        return focal

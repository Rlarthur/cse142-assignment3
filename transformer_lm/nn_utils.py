"""Numerically-stable softmax, activation functions, and cross-entropy loss."""

from __future__ import annotations

import math

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Numerically stable softmax.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension along which to compute softmax.

    Returns:
        Tensor of the same shape summing to 1 along ``dim``.
    """
    max = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - max
    exp = x_shifted.exp()
    sum = exp.sum(dim=dim, keepdim=True)
    return exp / sum

def silu(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid Linear Unit (SiLU / Swish) activation.

    Args:
        x: Input tensor of arbitrary shape.

    Returns:
        Tensor of the same shape.
    """
    sigmoid = 1 / (1 + (-x).exp())
    return x * sigmoid


def cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor,
) -> torch.Tensor:
    """Token-level cross-entropy loss (numerically stable).

    Args:
        logits: ``(B, T, V)`` — raw scores.
        targets: ``(B, T)`` — ground-truth token IDs.

    Returns:
        Scalar mean cross-entropy loss.
    """
    reshaped_logits = logits.reshape(-1, logits.shape[-1])
    reshaped_targets = targets.reshape(-1)
    indexed_logits = reshaped_logits[torch.arange(reshaped_logits.shape[0]), reshaped_targets]
    max_logit = reshaped_logits.max(dim=-1, keepdim=True).values
    sub_logits = reshaped_logits - max_logit
    exp_logits = sub_logits.exp()
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    log_sum_exp = sum_exp.log() + max_logit.squeeze(-1)
    loss = log_sum_exp - indexed_logits
    return loss.mean()

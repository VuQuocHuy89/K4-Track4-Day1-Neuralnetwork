"""Khởi tạo optimizer, scheduler và gradient clipping cho pipeline huấn luyện.

Triển khai bằng `torch.optim.*` và `torch.nn.utils.clip_grad_norm_` theo các
quy ước optimizer trong README mục 5.

Quy tắc cập nhật (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer.

    Hỗ trợ SGD, SGD có momentum, Adam và AdamW. `weight_decay` của Adam
    được cộng vào gradient; AdamW áp dụng suy giảm trọng số tách biệt.
    """
    if name not in OPTIMIZERS:
        raise ValueError(f"optimizer phải thuộc {OPTIMIZERS}, nhận được {name!r}.")
    if lr <= 0:
        raise ValueError("lr phải lớn hơn 0.")
    if weight_decay < 0:
        raise ValueError("weight_decay không được âm.")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
    return torch.optim.AdamW(
        params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """Tạo scheduler cosine hoặc trả về None nếu không bật scheduler."""
    if name is None:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, int(total_steps)), **kwargs
        )
    raise ValueError("scheduler hỗ trợ hiện tại: None hoặc 'cosine'.")


def clip_gradients(params, max_norm: float | None) -> float:
    """Trả về chuẩn L2 toàn cục trước khi cắt gradient.

    Với mixed precision FP16, gradient cần được unscale trước khi gọi hàm.
    """
    params = list(params)
    if max_norm is not None and max_norm <= 0:
        raise ValueError("max_norm phải lớn hơn 0 hoặc là None.")
    gradients = [parameter.grad.detach() for parameter in params if parameter.grad is not None]
    if not gradients:
        return 0.0
    per_parameter_norms = torch.stack(
        [torch.linalg.vector_norm(gradient.float(), ord=2) for gradient in gradients]
    )
    total_norm = torch.linalg.vector_norm(per_parameter_norms, ord=2)
    norm_value = float(total_norm.item())
    if max_norm is not None and torch.isfinite(total_norm):
        torch.nn.utils.clip_grad_norm_(params, float(max_norm))
    return norm_value

"""MLP cho bài toán CoverType theo kiến trúc bắt buộc của lab.

Model: MLP cho bài toán 7 lớp, shape cố định (xem README mục 3 và GUIDE, "Quy định kiến trúc"):

    x (B, 54) -> Linear(54, h1) -> ReLU -> [Dropout] -> Linear(h1, h2) -> ReLU -> [Dropout]
              -> ... -> Linear(h_last, 7) -> logits (B, 7)

Quy tắc:
  - Lớp cuối ra logit thô, KHÔNG softmax trong model (softmax nằm trong hàm mất mát).
  - Dropout chỉ đặt sau ReLU của lớp ẩn; không đặt trên đầu vào hay logit.
  - Mọi nn.Linear đều có bias. Không BatchNorm, không residual.
  - Số tham số phải khớp EXPECTED_PARAMS bên dưới.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Số tham số bắt buộc ứng với từng kiến trúc (in_features=54, num_classes=7)
EXPECTED_PARAMS = {
    (256, 128): 47_879,        # M-base  (baseline)
    (512, 256): 161_287,       # M-wide  (tuỳ chọn)
    (256, 128, 64): 55_687,    # M-deep  (tuỳ chọn)
}


class MLP(nn.Module):
    """MLP theo quy định ở đầu file.

    Args:
        hidden:   tuple số nơ-ron các lớp ẩn, ví dụ (256, 128)
        dropout:  xác suất TẮT nơ-ron q (nn.Dropout dùng p chính là xác suất tắt); 0.0 = không dùng
        init:     "zeros" | "normal" | "xavier" | "he" | "default"
    """

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        hidden = tuple(int(width) for width in hidden)
        if not hidden or any(width <= 0 for width in hidden):
            raise ValueError("hidden phải chứa các kích thước lớp ẩn dương.")
        if in_features != 54 or num_classes != 7:
            raise ValueError("Lab này yêu cầu đúng 54 đầu vào và 7 lớp đầu ra.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout phải nằm trong khoảng [0, 1).")

        layers = []
        previous_width = in_features
        for width in hidden:
            layers.extend((nn.Linear(previous_width, width), nn.ReLU()))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            previous_width = width
        layers.append(nn.Linear(previous_width, num_classes))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden
        self.dropout = float(dropout)
        self.init_name = init
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 54) float32  ->  logits: (B, 7) float32."""
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Khởi tạo tham số của mọi nn.Linear; bias được đặt bằng 0.

    init:
        "zeros"   : W = 0
        "normal"  : W ~ N(0, 0.01^2)
        "xavier"  : nn.init.xavier_normal_ (Var = 2/(n_in+n_out))
        "he"      : nn.init.kaiming_normal_(w, nonlinearity="relu")  (Var = 2/n_in)
        "default" : không làm gì (giữ khởi tạo mặc định của nn.Linear; KHÔNG phải He)
    """
    valid = {"zeros", "normal", "xavier", "he", "default"}
    if init not in valid:
        raise ValueError(f"init phải thuộc {sorted(valid)}, nhận được {init!r}.")
    if init == "default":
        return

    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, nn.Linear):
                continue
            if init == "zeros":
                nn.init.zeros_(module.weight)
            elif init == "normal":
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif init == "xavier":
                nn.init.xavier_normal_(module.weight)
            elif init == "he":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def count_params(model: nn.Module) -> int:
    """Đếm tổng số tham số có gradient được bật trong model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Trả về std của kích hoạt sau mỗi ReLU ẩn và sau lớp logits cuối."""
    model.eval()
    h = x
    stds = []
    layers = list(model.net)
    for index, layer in enumerate(layers):
        h = layer(h)
        if isinstance(layer, nn.ReLU) or index == len(layers) - 1:
            stds.append(float(h.std(unbiased=False).item()))
    return stds

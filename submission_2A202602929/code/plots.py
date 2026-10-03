"""Vẽ đường cong huấn luyện và ảnh so sánh các thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ loss, validation metrics và gradient norm theo epoch, kèm cấu hình chạy."""
    history = result.get("history", {})
    summary = result.get("summary", {})
    cfg = result.get("cfg", {})
    epochs = history.get("epoch", [])
    if not epochs:
        raise ValueError("result không có history để vẽ.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(epochs, history.get("train_loss", []), label="train loss")
    axes[0].plot(epochs, history.get("val_loss", []), label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, history.get("val_acc", []), label="val accuracy")
    if history.get("val_macro_f1"):
        axes[1].plot(epochs, history["val_macro_f1"], label="val macro-F1")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(epochs, history.get("grad_norm", []), label="grad norm (pre-clip)")
    axes[2].set_title("Gradient norm")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("L2 norm")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    best_epoch = summary.get("best_epoch")
    if best_epoch is not None:
        for axis in axes:
            axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.45)

    title = (
        f"{cfg.get('exp_id', 'experiment')} | {cfg.get('optimizer', '')} "
        f"lr={cfg.get('lr', '')} batch={cfg.get('batch', '')} "
        f"epochs={cfg.get('epochs', '')} hidden={cfg.get('hidden', '')} "
        f"dropout={cfg.get('dropout', '')} precision={cfg.get('precision', '')}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    if not results:
        raise ValueError("Cần ít nhất một result để vẽ.")
    fig, axis = plt.subplots(figsize=(8, 5))
    plotted = 0
    for result in results:
        history = result.get("history", {})
        values = history.get(metric)
        epochs = history.get("epoch", [])
        if not values or not epochs or len(values) != len(epochs):
            continue
        axis.plot(epochs, values, marker="o", markersize=3, label=result.get("cfg", {}).get("exp_id", "experiment"))
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        raise ValueError(f"Không có lịch sử {metric!r} để vẽ.")
    axis.set_title(title or f"So sánh {metric}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

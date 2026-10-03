"""Nạp, chia, chuẩn hoá và tạo batch cho dữ liệu CoverType.

Đầu vào là các tệp `data/processed/train.npz` và `eval.npz` do
`scripts/split_data.py` tạo.

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ tệp .npz theo quy ước dữ liệu ở đầu file."""
    root = Path(processed_dir)
    train_path = root / "train.npz"
    eval_path = root / "eval.npz"
    if not train_path.is_file() or not eval_path.is_file():
        raise FileNotFoundError(
            f"Thiếu dữ liệu đã chia: cần có {train_path} và {eval_path}."
        )

    with np.load(train_path, allow_pickle=False) as train_file:
        X_train = np.asarray(train_file["X"], dtype=np.float32)
        y_train = np.asarray(train_file["y"], dtype=np.int64)
    with np.load(eval_path, allow_pickle=False) as eval_file:
        X_eval = np.asarray(eval_file["X"], dtype=np.float32)
        y_eval = np.asarray(eval_file["y"], dtype=np.int64)
        eval_row_id = np.asarray(eval_file["row_id"], dtype=np.int64)

    assert X_train.ndim == X_eval.ndim == 2
    assert X_train.shape[1] == X_eval.shape[1] == 54
    assert y_train.shape == (len(X_train),) and y_eval.shape == (len(X_eval),)
    assert eval_row_id.shape == (len(X_eval),)
    assert y_train.min() >= 0 and y_train.max() <= 6
    assert y_eval.min() >= 0 and y_eval.max() <= 6
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation có phân tầng theo nhãn từ tập train.

    Trả về `X_tr, X_val, y_tr, y_val`; `seed` cố định kết quả phân tách.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction phải nằm trong khoảng (0, 1).")
    return train_test_split(
        X,
        y,
        test_size=val_fraction,
        stratify=y,
        random_state=seed,
    )


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu trên tập train sau khi tách val."""
    X_tr = np.asarray(X_tr, dtype=np.float32)
    if X_tr.ndim != 2 or X_tr.shape[1] < N_NUMERIC:
        raise ValueError(f"X_tr phải có ít nhất {N_NUMERIC} cột số.")
    mean = X_tr[:, :N_NUMERIC].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_tr[:, :N_NUMERIC].std(axis=0, dtype=np.float64).astype(np.float32)
    # Cột có độ lệch chuẩn 0 được giữ nguyên sau phép trừ mean.
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Hàm giữ nguyên đầu vào và xử lý độ lệch chuẩn bằng 0.
    """
    X_out = np.array(X, dtype=np.float32, copy=True)
    if X_out.ndim != 2 or X_out.shape[1] < N_NUMERIC:
        raise ValueError(f"X phải có ít nhất {N_NUMERIC} cột số.")
    safe_std = np.asarray(std, dtype=np.float32).copy()
    safe_std[safe_std == 0] = 1.0
    X_out[:, :N_NUMERIC] = (X_out[:, :N_NUMERIC] - mean) / safe_std
    return X_out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Thống kê chuẩn hoá chỉ lấy từ train; cùng mean/std được áp dụng lên train,
    validation và eval. Tensor đặc trưng là float32, nhãn là int64. Hàm cũng
    báo kích thước các tập và accuracy của baseline đoán lớp đa số trên val.
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, X_val, y_tr, y_val = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )

    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    result = {
        "X_tr": torch.as_tensor(X_tr, dtype=torch.float32, device=device),
        "y_tr": torch.as_tensor(y_tr, dtype=torch.int64, device=device),
        "X_val": torch.as_tensor(X_val, dtype=torch.float32, device=device),
        "y_val": torch.as_tensor(y_val, dtype=torch.int64, device=device),
        "X_eval": torch.as_tensor(X_eval, dtype=torch.float32, device=device),
        "y_eval": torch.as_tensor(y_eval, dtype=torch.int64, device=device),
        "eval_row_id": eval_row_id,
        "standardizer_mean": mean,
        "standardizer_std": std,
    }

    majority_class = int(np.bincount(y_tr, minlength=7).argmax())
    val_majority_accuracy = float(np.mean(y_val == majority_class))
    print(
        f"train: {len(y_tr):,}; val: {len(y_val):,}; eval: {len(y_eval):,}; "
        f"majority class from train: {majority_class}; val accuracy: {val_majority_accuracy:.4f}"
    )
    return result


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Batch cuối có thể có ít mẫu hơn `batch_size`.
    """
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")
    n_samples = len(X)
    if shuffle:
        indices = torch.randperm(
            n_samples, generator=generator, device=X.device
        )
    else:
        indices = torch.arange(n_samples, device=X.device)
    for start in range(0, n_samples, batch_size):
        batch_indices = indices[start : start + batch_size]
        yield X[batch_indices], y[batch_indices]

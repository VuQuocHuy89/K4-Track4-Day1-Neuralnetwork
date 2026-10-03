"""Đánh giá, huấn luyện thí nghiệm, dự đoán và ghi file nộp.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mỗi thí nghiệm được biểu diễn bằng một dict cấu hình truyền vào `run_experiment`.

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

# Cấu hình mặc định của baseline M-base; learning rate được chọn bằng validation trong notebook.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # notebook chọn learning rate bằng validation
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
    betas=(0.9, 0.999),
    eps=1e-8,
    scheduler=None,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm phải là ma trận vuông.")
    true_positive = np.diag(cm)
    predicted = cm.sum(axis=0)
    actual = cm.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        actual,
        out=np.zeros_like(true_positive),
        where=actual > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn int64 (N,) bằng argmax của logits theo từng lô."""
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")
    model.eval()
    predictions = []
    for start in range(0, len(X), batch_size):
        logits = model(X[start : start + batch_size])
        predictions.append(logits.argmax(dim=1))
    if not predictions:
        return torch.empty(0, dtype=torch.int64, device=X.device)
    return torch.cat(predictions, dim=0).to(dtype=torch.int64)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Tính loss, accuracy, macro-F1 và ma trận nhầm lẫn ở chế độ eval."""
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")
    model.eval()
    n_samples = len(X)
    if n_samples == 0:
        raise ValueError("Không thể đánh giá tập rỗng.")

    total_loss = 0.0
    confusion = torch.zeros((7, 7), dtype=torch.int64, device=X.device)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        logits = model(X[start:stop])
        labels = y[start:stop].to(dtype=torch.int64)
        loss = compute_loss(logits, labels, loss_name)
        total_loss += float(loss.item()) * (stop - start)
        predictions = logits.argmax(dim=1)
        flat_indices = labels * 7 + predictions
        confusion += torch.bincount(flat_indices, minlength=49).reshape(7, 7)

    count = confusion.sum().item()
    accuracy = float(torch.diagonal(confusion).sum().item() / count)
    macro_f1 = macro_f1_from_confusion(confusion.cpu().numpy())
    return {
        "loss": total_loss / n_samples,
        "acc": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion.cpu().numpy(),
    }


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy trên logit thô và nhãn int64.
       "mse" : mean squared error giữa logit và one-hot của nhãn.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y.to(dtype=torch.int64))
    if loss_name == "mse":
        one_hot = F.one_hot(y.to(dtype=torch.int64), num_classes=logits.shape[1])
        return F.mse_loss(logits, one_hot.to(dtype=logits.dtype), reduction="mean")
    raise ValueError("loss_name phải là 'ce' hoặc 'mse'.")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Huấn luyện trên train, theo dõi validation sau mỗi epoch và lưu state có
    validation loss thấp nhất. Kết quả gồm lịch sử theo epoch, các metric tốt
    nhất trên validation, thời gian chạy, bộ nhớ GPU và trạng thái phân kỳ.
    Tập eval không tham gia lựa chọn epoch hoặc cấu hình.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    cfg["hidden"] = tuple(int(width) for width in cfg["hidden"])
    if cfg["lr"] is None or not math.isfinite(float(cfg["lr"])) or cfg["lr"] <= 0:
        raise ValueError("Cần chọn learning rate dương bằng validation trước khi huấn luyện.")
    if int(cfg["batch"]) <= 0 or int(cfg["epochs"]) <= 0:
        raise ValueError("batch và epochs phải lớn hơn 0.")
    if cfg["hidden"] not in EXPECTED_PARAMS:
        raise ValueError(f"Kiến trúc hidden={cfg['hidden']} không có trong quy định của lab.")
    if cfg["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision phải là 'fp32', 'fp16' hoặc 'bf16'.")

    set_seed(int(cfg["seed"]))
    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    device = X_tr.device
    if cfg["precision"] == "fp16" and device.type != "cuda":
        raise ValueError("FP16 trong lab này cần CUDA.")
    if cfg["precision"] == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("GPU hiện tại không hỗ trợ BF16.")

    model = MLP(
        hidden=cfg["hidden"], dropout=float(cfg["dropout"]), init=cfg["init"]
    ).to(device)
    if count_params(model) != EXPECTED_PARAMS[cfg["hidden"]]:
        raise AssertionError(
            f"Số tham số sai: {count_params(model)} != {EXPECTED_PARAMS[cfg['hidden']]}"
        )
    optimizer = build_optimizer(
        cfg["optimizer"],
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        momentum=float(cfg["momentum"]),
        betas=tuple(cfg["betas"]),
        eps=float(cfg["eps"]),
    )

    steps_per_epoch = math.ceil(len(X_tr) / int(cfg["batch"]))
    scheduler = build_scheduler(
        optimizer,
        cfg.get("scheduler"),
        total_steps=steps_per_epoch * int(cfg["epochs"]),
        **cfg.get("scheduler_kwargs", {}),
    )
    precision_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        cfg["precision"]
    )
    use_scaler = cfg["precision"] == "fp16"
    if use_scaler:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    clip_events = 0
    finite_gradient_steps = 0

    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg["seed"]))
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }
    step0_loss = float(evaluate(model, X_val, y_val, cfg["loss"])["loss"])
    best_val_loss = math.inf
    best_state = None
    best_epoch = None
    best_train_loss = None
    best_val_acc = None
    best_val_macro_f1 = None
    diverged = not math.isfinite(step0_loss)
    final_train_loss = None
    final_val_loss = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, int(cfg["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        grad_norms = []
        epoch_failed = False

        if not diverged:
            for xb, yb in iterate_batches(
                X_tr, y_tr, int(cfg["batch"]), generator=generator, shuffle=True
            ):
                optimizer.zero_grad(set_to_none=True)
                if precision_dtype is None:
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
                else:
                    with torch.autocast(
                        device_type=device.type,
                        dtype=precision_dtype,
                        enabled=True,
                    ):
                        logits = model(xb)
                        loss = compute_loss(logits, yb, cfg["loss"])

                if not bool(torch.isfinite(loss).item()):
                    diverged = True
                    epoch_failed = True
                    break

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                grad_norm = clip_gradients(model.parameters(), cfg["clip_norm"])
                if not math.isfinite(grad_norm):
                    if scaler is None:
                        diverged = True
                        epoch_failed = True
                        optimizer.zero_grad(set_to_none=True)
                        break
                    grad_norms.append(grad_norm)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)  # GradScaler skips this update after detecting inf/nan.
                    scaler.update()
                    if scheduler is not None and scaler.get_scale() >= old_scale:
                        scheduler.step()
                    continue
                finite_gradient_steps += 1
                if cfg["clip_norm"] is not None and grad_norm > float(cfg["clip_norm"]):
                    clip_events += 1
                grad_norms.append(grad_norm)

                if scaler is not None:
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None and scaler.get_scale() >= old_scale:
                        scheduler.step()
                else:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

        if epoch_failed:
            train_loss = val_loss = val_acc = val_macro_f1 = float("nan")
        elif not diverged:
            train_metrics = evaluate(model, X_tr, y_tr, cfg["loss"])
            val_metrics = evaluate(model, X_val, y_val, cfg["loss"])
            train_loss = float(train_metrics["loss"])
            val_loss = float(val_metrics["loss"])
            val_acc = float(val_metrics["acc"])
            val_macro_f1 = float(val_metrics["macro_f1"])
            if not all(math.isfinite(value) for value in (train_loss, val_loss, val_acc, val_macro_f1)):
                diverged = True
                train_loss = val_loss = val_acc = val_macro_f1 = float("nan")
        else:
            train_loss = val_loss = val_acc = val_macro_f1 = float("nan")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - epoch_start
        mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else float("nan")
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["grad_norm"].append(mean_grad_norm)
        history["epoch_time_s"].append(epoch_time)

        if math.isfinite(val_loss):
            final_train_loss = train_loss
            final_val_loss = val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_train_loss = train_loss
                best_val_acc = val_acc
                best_val_macro_f1 = val_macro_f1
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

        print(
            f"{cfg['exp_id']} | epoch {epoch:02d}/{cfg['epochs']} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"val_macro_f1={val_macro_f1:.5f} grad_norm={mean_grad_norm:.5g} "
            f"time={epoch_time:.2f}s" + (" DIVERGED" if diverged else "")
        )
        if diverged:
            break

    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    else:
        peak_mem_mb = None
    completed_times = [value for value in history["epoch_time_s"] if math.isfinite(value)]
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": None if best_epoch is None else best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "val_acc": best_val_acc,
        "val_macro_f1": best_val_macro_f1,
        "time_per_epoch_s": float(np.mean(completed_times)) if completed_times else None,
        "peak_mem_MB": peak_mem_mb,
        "clip_events": clip_events,
        "finite_gradient_steps": finite_gradient_steps,
        "clip_fraction": (
            clip_events / finite_gradient_steps if finite_gradient_steps else None
        ),
        "diverged": bool(diverged),
        "best_train_loss": best_train_loss,
    }
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id = np.asarray(row_id)
    preds = np.asarray(preds)
    if row_id.ndim != 1 or preds.ndim != 1 or len(row_id) != len(preds):
        raise ValueError("row_id và preds phải là hai mảng một chiều cùng độ dài.")
    if not np.issubdtype(row_id.dtype, np.integer):
        if not np.all(np.equal(row_id, row_id.astype(np.int64))):
            raise ValueError("row_id phải chứa số nguyên.")
    if not np.issubdtype(preds.dtype, np.integer):
        if not np.all(np.equal(preds, preds.astype(np.int64))):
            raise ValueError("pred phải chứa số nguyên.")
    preds = preds.astype(np.int64)
    if len(np.unique(row_id)) != len(row_id):
        raise ValueError("row_id bị trùng.")
    if len(preds) and ((preds < 0).any() or (preds > 6).any()):
        raise ValueError("pred phải nằm trong khoảng 0..6.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("row_id", "pred"))
        writer.writerows(zip(row_id.astype(np.int64), preds))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Nạp state tốt nhất, dự đoán toàn bộ eval và ghi CSV theo định dạng evaluator."""
    if result.get("best_state") is None:
        raise ValueError("Lần chạy không có best_state hợp lệ; không thể tạo dự đoán eval.")
    device = data["X_eval"].device
    model = MLP(
        hidden=tuple(cfg["hidden"]),
        dropout=float(cfg["dropout"]),
        init=cfg["init"],
    ).to(device)
    model.load_state_dict(result["best_state"])
    predictions = predict(model, data["X_eval"]).detach().cpu().numpy()
    write_predictions(data["eval_row_id"], predictions, pred_path)
    print(f"Đã ghi {len(predictions):,} dự đoán: {pred_path}")

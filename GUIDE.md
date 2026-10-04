# GUIDE — Lab Day 1: từng bước

Đọc [`README.md`](README.md) (luật chơi, sản phẩm nộp) và [`RUBRIC.md`](RUBRIC.md) trước. Tài liệu này chỉ **gợi ý và nêu tiêu chí kiểm tra**, không đưa code hoàn chỉnh. Bạn tự viết code.

**Số lượng thí nghiệm là tuỳ bạn.** Part 3 là *menu gợi ý*, không phải danh sách bắt buộc. Điểm dựa vào chất lượng thiết kế thí nghiệm, độ phủ chủ đề và chất lượng kết luận (xem RUBRIC), không dựa vào việc chạy cho đủ N lần.

**Về các con số tham chiếu:** những mốc sau chắc chắn đúng về mặt toán/dữ liệu: loss bước 0 ≈ ln 7 ≈ 1,946; "đoán luôn lớp 1 (nhãn gốc 2)" cho accuracy 0,4876 và macro-F1 chỉ ≈ 0,094 trên eval. Ngoài ra giảng viên đã chạy vài lần tham chiếu (1 seed, 1 máy) để đặt mức điểm eval, xem mục *Cách đánh giá* bên dưới. Đó là mốc để bạn biết mình đang ở đâu, không phải đáp án.

---

## Quy định kiến trúc mô hình (bắt buộc, mọi sinh viên giống nhau)

Bạn **tự định nghĩa model** (một class `nn.Module` do bạn viết). Không dùng mô hình dựng sẵn/pretrained hay `sklearn.MLPClassifier`. Shape được cố định để kết quả so sánh được giữa các bạn.

**Đầu vào / đầu ra (mọi mô hình):**

| | Shape | Kiểu |
|---|---|---|
| `x` (lô đầu vào) | `(B, 54)` | `float32` |
| `y` (nhãn) | `(B,)` | `int64`, giá trị `0..6` |
| `logits` (đầu ra mô hình) | `(B, 7)` | điểm số thô, **không** softmax trong model |

**Các kiến trúc (MLP, ReLU ở mọi lớp ẩn, có bias ở mọi lớp tuyến tính):**

| Tên | Dùng ở | Các lớp (số nơ-ron) | Số tham số |
|---|---|---|---|
| `M-base` | **Bắt buộc**: baseline | `54 → 256 → 128 → 7` | 47 879 |
| `M-wide` | Tuỳ chọn: thí nghiệm độ rộng | `54 → 512 → 256 → 7` | 161 287 |
| `M-deep` | Tuỳ chọn: thí nghiệm độ sâu | `54 → 256 → 128 → 64 → 7` | 55 687 |

Luồng tensor của `M-base` (`B` = batch):

```
x (B,54) → Linear(54,256) → (B,256) → ReLU → [Dropout] → Linear(256,128) → (B,128)
        → ReLU → [Dropout] → Linear(128,7) → logits (B,7)
```

Quy tắc:
- `[Dropout]` chỉ đặt **sau ReLU của lớp ẩn**, không đặt trên đầu vào hay trên logit; tắt (q = 0) khi không làm thí nghiệm dropout.
- Không BatchNorm/LayerNorm, không kết nối tắt (residual). Mỗi lần chỉ đổi đúng yếu tố của thí nghiệm.
- Lớp cuối ra logit thô; softmax nằm trong hàm mất mát.
- Bắt buộc `assert sum(p.numel() for p in model.parameters()) == <số ở bảng trên>` ngay sau khi tạo model.

---

## Cách đánh giá (evaluation)

**Hai tập, cố định cho mọi sinh viên** (xem `data/README.md`):

| Tập | Số mẫu | Dùng để |
|---|---|---|
| `train` | 464 809 | huấn luyện; bạn tự tách một phần làm **validation** để chọn cấu hình và dừng sớm |
| `eval` | 116 203 | **chỉ để chấm điểm cuối.** Không dùng để chọn lr, epoch, kiến trúc, kỹ thuật hay cách chuẩn hoá |

Cả hai chia phân tầng theo nhãn bằng `data/split_metadata.csv` (seed 42). Bạn không tự chia lại tập train/eval.

**Quy trình chuẩn:**
1. Chạy `python scripts/split_data.py` để tạo `data/processed/train.npz` và `eval.npz`.
2. Trong code, tách validation từ `train` (khuyến nghị 20%, phân tầng, seed 42). Chuẩn hoá bằng thống kê của phần train còn lại.
3. Mọi thí nghiệm so sánh bằng **val** (val loss, val accuracy, val macro-F1).
4. Chọn cấu hình cuối cùng **chỉ bằng val**. Huấn luyện/chọn epoch bằng val; sau đó dự đoán trên **toàn bộ eval** ở chế độ `eval()` (không dropout), ghi `predictions_eval.csv` (cột `row_id,pred`, nhãn 0..6).
5. Chạy `python scripts/evaluate.py --pred predictions_eval.csv --out eval_result.json`. Script kiểm tra file hợp lệ rồi in accuracy, **macro-F1**, precision/recall/F1 từng lớp và ma trận nhầm lẫn. Giảng viên chấm bằng đúng script này.

**Chỉ số:**
- **macro-F1 (chỉ số chính):** trung bình cộng F1 của 7 lớp, mỗi lớp trọng số như nhau. Dữ liệu mất cân bằng nên lớp hiếm (nhãn 3 chỉ ~0,5%) ảnh hưởng ngang lớp lớn. Đoán luôn lớp đa số chỉ đạt ≈ 0,094.
- **accuracy (phụ):** tỉ lệ dự đoán đúng; đoán luôn lớp đa số đạt 0,4876.
- Công thức: `P_c = TP/(TP+FP)`, `R_c = TP/(TP+FN)`, `F1_c = 2·P_c·R_c/(P_c+R_c)` (bằng 0 nếu mẫu số bằng 0).

**Mốc tham chiếu của giảng viên** (1 seed, một máy; `M-base`, 20 epoch, SGD+momentum hoặc Adam, lr khác nhau; kết quả trên eval):
- 5 epoch, lr rất nhỏ: macro-F1 ≈ 0,57.
- 20 epoch, lr vừa phải: ≈ 0,83 đến 0,87 (các cấu hình này chưa hội tụ hẳn, epoch tốt nhất đều là cuối).
- mạng lớn hơn (3 lớp ẩn 512-256-128) + 40 epoch: ≈ 0,90.
- Chênh lệch giữa val và eval nhỏ (≤ ~0,005 trong các lần chạy này), nên val là ước lượng đáng tin của eval.
Mức điểm theo macro-F1 nằm trong `RUBRIC.md` (mục 7).

**Phân tích lỗi (cũng được chấm):** nhìn F1 từng lớp và ma trận nhầm lẫn: lớp nào khó nhất, nhầm với lớp nào, vì sao (số mẫu, tương đồng đặc trưng).

---

## Code khung (pseudo-code) trong `code/`

Thư mục `code/` có sẵn khung code: **mọi hàm chỉ có docstring, các bước gợi ý và `raise NotImplementedError`. Bạn phải tự hoàn thiện.**

| File | Phần của bạn | Dùng ở |
|---|---|---|
| `data.py` | nạp train/eval, tách val, chuẩn hoá, đưa lên device, chia lô | Part 0 |
| `model.py` | class `MLP`, khởi tạo, đếm tham số, thống kê kích hoạt | Part 1 |
| `optimizer.py` | chọn bộ tối ưu (`torch.optim`), cắt gradient | Part 2–3 |
| `train.py` | `evaluate`, `predict`, `run_experiment`, ghi `predictions_eval.csv` | Part 2–4 |
| `plots.py` | ảnh từng thí nghiệm và ảnh chồng | Part 2–4 |
| `results_table.py` | lưu JSON, điền `experiments.xlsx` | Part 4 |
| `lab.ipynb` | notebook khung: các mục Part 0–4 với ô `# TODO` | tất cả |

Bạn được đổi chữ ký hàm và thêm file nếu cần, miễn là sản phẩm nộp đúng README mục 6 và không còn `NotImplementedError`.

---

## Part 0 — Chuẩn bị dữ liệu

Khung code: `code/data.py`.

1. Bật GPU, in ra `torch.__version__` và tên GPU (`torch.cuda.get_device_name(0)`; trên Mac có thể dùng `mps`).
2. Viết `set_seed(s)` đặt seed cho `random`, `numpy`, `torch`, `torch.cuda`.
3. **Chia dữ liệu bằng metadata đã cho:** từ thư mục gốc repo chạy
   ```bash
   python scripts/split_data.py
   ```
   Script đọc `data/covtype.csv.gz` và `data/split_metadata.csv`, kiểm tra tính toàn vẹn, rồi tạo `data/processed/train.npz` (464 809 mẫu) và `data/processed/eval.npz` (116 203 mẫu). Nhãn đã được đổi về `0..6` kiểu `int64`; đặc trưng là `float32`. **Đừng sửa metadata.**
4. **Tách validation từ train** (không đụng eval): 20% của train, phân tầng theo nhãn, seed 42 (`train_test_split(..., stratify=y, random_state=42)`). Dùng cùng cách tách cho mọi thí nghiệm.
5. **Chuẩn hoá:** chỉ 10 cột đầu (số liên tục). Tính trung bình/độ lệch chuẩn **chỉ trên phần train còn lại** (sau khi tách val), áp dụng cho val và eval. 44 cột nhị phân giữ nguyên.
   - Vì sao không được tính trên val hay eval? (Gợi ý: rò rỉ thông tin.)
6. Đưa toàn bộ `X_tr, y_tr, X_val, y_val, X_eval, y_eval` lên GPU dưới dạng tensor một lần. **Không cần `DataLoader`**: tự xáo `torch.randperm(N)` mỗi epoch rồi cắt lô. Cách này nhanh hơn nhiều khi chạy nhiều cấu hình.

**Tự kiểm tra Part 0:**
- [x] Kích thước: `train` = 464 809, `eval` = 116 203; sau khi tách val (20%) còn 371 847 mẫu train và 92 962 mẫu val.
- [x] Tỉ lệ lớp ở train, val, eval gần như bằng nhau.
- [x] Trung bình/độ lệch chuẩn của 10 cột số trên phần *train còn lại* ≈ 0 / 1.
- [x] In ra accuracy của chiến lược "luôn đoán lớp đa số" trên val (≈ 0,4876). Đó là mốc thấp nhất mà mô hình phải vượt.
- [x] Không có đoạn code nào đưa `X_eval` vào bước chuẩn hoá, chọn cấu hình hay dừng sớm.

---

## Part 1 — Định nghĩa model và kiểm tra "sức khoẻ" ban đầu

Khung code: `code/model.py`.

**Mục tiêu:** có một model đúng shape, đúng số tham số, và qua được các phép thử rẻ nhất của slide (Chương 5) *trước khi* huấn luyện lâu.

### 1a. Viết model
Class `MLP(nn.Module)` nhận `hidden` (danh sách số nơ-ron lớp ẩn, ví dụ `[256,128]`), `dropout` (q) và `init`. `forward(x)` trả logits `(B,7)`. Ghép từ `nn.Linear`, ReLU, `nn.Dropout` (hoặc `nn.Sequential`) tuỳ bạn, miễn đúng mục *Quy định kiến trúc*.

- Dropout của PyTorch dùng `p` là xác suất **tắt**; slide dùng chữ `q` cho cùng ý nghĩa.
- Khởi tạo mặc định của `nn.Linear` *không phải* He (slide, Chương 4). Muốn baseline dùng He thì phải tự gọi `nn.init` (xem Part 3, nhóm khởi tạo). Ghi đúng cách khởi tạo bạn dùng vào bảng.
- Hàm mất mát: `nn.CrossEntropyLoss` / `F.cross_entropy` nhận **logit thô** và nhãn `int64`. Đừng đặt softmax trong model.

### 1b. Kiểm tra shape và số tham số
- `assert` số tham số khớp bảng (`M-base` = 47 879).
- Cho một lô ngẫu nhiên `(8, 54)` chạy qua model; in shape sau từng lớp và kiểm tra đầu ra `(8, 7)`.

### 1c. Hai phép thử "sức khoẻ" của slide
1. **Loss bước 0:** với model vừa khởi tạo (chế độ `eval`), loss cross-entropy trên tập val phải gần `ln 7 ≈ 1,946`. In giá trị đo được. Lệch nhiều thì xem bảng "triệu chứng" ở cuối GUIDE.
2. **Quá khớp một lô nhỏ:** lấy 20 mẫu, tắt dropout, huấn luyện vài trăm bước. Loss phải về **gần 0** (accuracy 100% trên 20 mẫu đó). Nếu không, gần như chắc chắn là lỗi code (nhãn lệch, softmax hai lần, quên `zero_grad`, tham số không được đưa vào optimizer), ít khi do năng lực mô hình.

### 1d. Gradient có chảy không?
Sau một lần `loss.backward()`, in chuẩn gradient của từng tham số (`W1,b1,W2,b2,W3,b3`). Tất cả phải khác `None` và khác 0. Đây là cách phát hiện sớm "gradient không chảy" (slide, Chương 5).

**Tự kiểm tra Part 1:**
- [x] Số tham số khớp bảng; logits có shape `(B,7)`.
- [x] Loss bước 0 ≈ 1,946.
- [x] Quá khớp 20 mẫu: loss → gần 0.
- [x] Mọi tham số có gradient khác 0 sau `backward()`.

---

## Part 2 — Pipeline huấn luyện và baseline

Khung code: `code/train.py`, `code/optimizer.py`, `code/plots.py`.

Bạn được dùng mọi thành phần PyTorch: `nn.init.*`, `torch.optim.*`, `clip_grad_norm_`, `autocast`, `GradScaler`, ... Phần bạn tự viết là pipeline.

### 2a. Hàm `run_experiment(cfg)`
Viết **một** hàm duy nhất `run_experiment(cfg) -> dict` nhận một dict cấu hình (ví dụ: loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout, clip_norm, precision, init, seed) và trả về lịch sử + tóm tắt. Mọi thí nghiệm sau chỉ là *đổi dict cấu hình*. Điều này vừa giảm lỗi vừa bảo đảm công bằng.

Trong mỗi epoch ghi lại:
- `train_loss`, **tính ở chế độ `eval()`** trên toàn bộ train (hoặc một tập con cố định, ví dụ 50 000 mẫu), để đường train và val *so sánh được với nhau* (loss trong lúc huấn luyện có dropout nên không cùng thang đo).
- `val_loss`, `val_acc`, `val_macro_f1`. Macro-F1: `sklearn.metrics.f1_score(average="macro")`.
- `grad_norm` trung bình trong epoch: chuẩn L2 toàn cục của gradient, **đo trước khi clip**. (`torch.nn.utils.clip_grad_norm_` trả về đúng giá trị này, trước khi cắt.)
- Thời gian mỗi epoch (nhớ `torch.cuda.synchronize()` trước khi đo) và bộ nhớ GPU cực đại (`torch.cuda.max_memory_allocated`).
- **Loss bước 0** trên tập val, đo *trước* bước cập nhật đầu tiên.
- Epoch có `val_loss` thấp nhất ("best epoch"); báo cáo metric ở epoch đó (hoạt động như dừng sớm, slide Chương 4).
- Cờ `diverged` nếu loss thành `NaN/inf`: dừng sớm và ghi lại, đừng để notebook treo.

Sau **mỗi** lần chạy, ghi lịch sử ra `results/<exp_id>.json` và lưu ảnh `figures/<exp_id>.png`. Vì notebook nằm trong `code/`, đường dẫn ghi là `../results/` và `../figures/` (xem README mục 6). Colab/Kaggle có thể ngắt kết nối; nếu kết quả đã nằm trên Drive hay `/kaggle/working` thì bạn không phải chạy lại.

Ảnh `figures/<exp_id>.png` gồm ít nhất 3 ô: (1) train và val loss theo epoch, (2) val accuracy (và macro-F1 nếu được), (3) `grad_norm`. Có tiêu đề ghi `exp_id` và cấu hình, có chú thích, có nhãn trục.

### 2b. Baseline (bắt buộc)
| Thành phần | Giá trị |
|---|---|
| Kiến trúc | `M-base`: `54 → 256 → 128 → 7`, ReLU |
| Khởi tạo | He |
| Mất mát | Cross-entropy |
| Bộ tối ưu | SGD + momentum 0,9 |
| Batch / Epoch | 512 / 20 |
| Dropout / Clip / Precision | 0 / không / FP32 |

Tìm `lr` cho baseline: chạy nhanh vài giá trị (ví dụ quanh 0,01 → 0,1) trên **tập val**, chọn một giá trị hợp lý.

**Khuyến khích mạnh:** chạy baseline với 2–3 seed khác nhau để đo **độ nhiễu**: dao động giữa các seed của val loss / accuracy / macro-F1. Chênh lệch giữa hai cấu hình nhỏ hơn ~2 lần độ lệch chuẩn này thì không phải là bằng chứng. Nếu bạn chỉ chạy một seed, hãy nêu rõ đó là hạn chế trong báo cáo.

**Tự kiểm tra Part 2:**
- [x] Loss bước 0 baseline ≈ 1,946.
- [x] Val accuracy cuối > 48,8% (mốc "đoán đa số"). Nếu không, có lỗi.
- [x] Đường train/val loss trông bình thường; bạn mô tả được hình dạng (còn giảm? bắt đầu quá khớp?).
- [x] Có file `results/*.json` và ảnh cho baseline.

---

## Part 3 — Thí nghiệm (menu gợi ý, tự chọn)

Bạn chọn thí nghiệm nào và bao nhiêu thí nghiệm tuỳ ý. Rubric thưởng cho **độ phủ** (bạn thử được bao nhiêu chủ đề trong 7 chủ đề dưới đây) và **chất lượng** (công bằng, có dự đoán, có giải thích cơ chế), không thưởng việc chạy thật nhiều lần.

**Cách làm mỗi thí nghiệm:**
1. **Dự đoán trước** (1–2 câu, ghi vào notebook): bạn nghĩ sẽ xảy ra gì và vì sao?
2. Đổi **một yếu tố** so với baseline, giữ nguyên mọi thứ còn lại (kể cả seed, số epoch).
3. Chạy, lưu ảnh và kết quả vào bảng.
4. **Đối chiếu**: khớp hay khác dự đoán? Vì sao?

**Lộ trình khi ít thời gian (gợi ý):** baseline (nhiều seed nếu được) → đổi bộ tối ưu (kèm vài lr) → mỗi chủ đề kỹ thuật còn lại một thí nghiệm có chủ đích.

**Đặt tên `exp_id`:** tự do nhưng nhất quán, ví dụ `base-s1`, `opt-adam-lr1e-3`, `drop-0.3`, `clip-1.0-highlr`.

### Chủ đề 1 — Hàm mất mát (`loss`)
- Cross-entropy vs MSE trên nhãn one-hot. Ghi rõ MSE tính thế nào (`nn.MSELoss` không có hệ số 1/2 và lấy trung bình trên mọi phần tử).
- Dự đoán: loss nào cho gradient lớn hơn khi dự đoán sai nặng? (Slide Chương 3: gradient không bão hoà của cross-entropy.)
- **Lưu ý:** giá trị loss của CE và MSE khác thang đo, nên **không so loss trực tiếp**; so accuracy, macro-F1 và tốc độ hội tụ.

### Chủ đề 2 — Bộ tối ưu hoá (`optimizer`)
- SGD, SGD+momentum, Adam, AdamW (`torch.optim`). Ghi đúng tham số: `momentum`, `betas`, `eps`, `weight_decay`.
- Mỗi bộ tối ưu **nên thử ≥ 2–3 giá trị lr** cách nhau 3–10 lần, rồi so **ở lr tốt nhất của mỗi bộ**. Nếu chỉ thử một lr thì "Adam thắng SGD" có nghĩa là gì? Gợi ý khởi điểm (chưa kiểm chứng): SGD quanh 0,01–0,1; Adam/AdamW quanh 1e-4–3e-3; AdamW `weight_decay=0,01`.
- Công thức cần hiểu (slide Chương 4):

| Bộ tối ưu | Cập nhật |
|---|---|
| SGD | `w ← w − η·g` |
| SGD + momentum (dạng PyTorch) | `v ← μ·v + g`;  `w ← w − η·v` |
| Adam | `m ← β₁m + (1−β₁)g`; `v ← β₂v + (1−β₂)g²`; `m̂ = m/(1−β₁ᵗ)`; `v̂ = v/(1−β₂ᵗ)`; `w ← w − η·m̂/(√v̂ + ε)` |
| AdamW | như Adam nhưng suy giảm trọng số tách riêng: `w ← w − ηλw − η·m̂/(√v̂ + ε)` |

- Adam và AdamW khác nhau ở đâu? Với `weight_decay=0` chúng cho cùng kết quả; bạn có thể kiểm chứng bằng một lần chạy.

### Chủ đề 3 — Hyper-parameter (`hparam`)
Gợi ý: `lr`, `batch size` (ví dụ 128 / 512 / 2048), độ rộng (`M-wide`), độ sâu (`M-deep`), `weight_decay`, số epoch.
- Cùng số epoch nhưng batch khác nhau nghĩa là **số bước cập nhật khác nhau**; nhắc điều này khi giải thích.
- Batch lớn có cần tăng lr không? Slide Chương 4 nêu *quy tắc tăng lr theo lô* (lô ×k thì η ×k, kèm khởi động). Thử và nhận xét.
- Ghi thời gian mỗi epoch.

### Chủ đề 4 — Dropout (`dropout`)
- `q ∈ {0,1; 0,3; 0,5}` (hoặc các giá trị bạn chọn). Quan sát khoảng cách train–val loss và val accuracy.
- Dự đoán: dropout có chắc giúp ích không nếu mô hình *chưa* quá khớp? (Xem bảng "triệu chứng": dropout là thuốc cho quá khớp.)
- Khi vẽ, train loss phải đo ở `eval()` (xem 2a), nếu không bạn sẽ thấy dropout "làm tăng train loss" chỉ vì còn bị tắt nơ-ron.

### Chủ đề 5 — Cắt gradient (`clipping`)
Công thức (slide): `g ← g · min(1, c/‖g‖)` với `‖g‖` là chuẩn L2 **toàn cục** của mọi gradient gộp lại. Dùng `torch.nn.utils.clip_grad_norm_(model.parameters(), c)`.
- **Nhìn `grad_norm` của baseline trước.** Chọn `c` sao cho clipping *thực sự* kích hoạt ở một phần các bước. Nếu `c` lớn hơn mọi `grad_norm` thì clipping không làm gì cả và chênh lệch chỉ là nhiễu.
- **Thí nghiệm phản chứng (nên làm):** tăng lr (×10 hoặc hơn) tới khi huấn luyện không clip bị dao động/NaN. Chạy cùng lr đó **có clip** và **không clip**. Đây mới là tình huống clipping dùng cho: gradient đột ngột lớn.
- Ghi `grad_norm` trước khi clip để thấy các "gai".

### Chủ đề 6 — Mixed precision (`amp`)
- FP16: `torch.autocast("cuda", dtype=torch.float16)` + `torch.amp.GradScaler("cuda")`. Nếu có clipping, gọi `scaler.unscale_(optimizer)` trước khi clip.
- BF16: `autocast(dtype=torch.bfloat16)`, thường không cần GradScaler (slide).
- Tham số vẫn FP32; autocast chỉ hạ độ chính xác của phép toán. Giải thích vì sao FP16 cần nhân loss với hệ số `s` còn BF16 thường không (so sánh khoảng giá trị biểu diễn của FP16 và BF16).
- Kiểm tra `torch.cuda.is_bf16_supported()`. Một số GPU (như T4, P100) không có phần cứng BF16 nên có thể chậm hoặc không hỗ trợ. Nếu không chạy được, ghi rõ lý do trong bảng (`notes`).
- Đo: thời gian mỗi epoch, bộ nhớ cực đại, độ chính xác so với FP32. **Chỉ báo "nhanh hơn" nếu số đo cho thấy vậy.** Với mạng nhỏ, thời gian thường bị chi phối bởi chi phí gọi kernel, nên có thể không nhanh hơn; đó vẫn là kết quả hợp lệ, hãy giải thích. (Tuỳ chọn: thử một mạng rất rộng để xem khi nào mixed precision có lợi.)

### Chủ đề 7 — Khởi tạo tham số (`init`)
Áp dụng bằng `torch.nn.init` (bias = 0):

| Tên | Công thức | PyTorch |
|---|---|---|
| `zeros` | `W = 0` | `nn.init.zeros_` |
| `normal` | `W ~ N(0, 0,01²)` | `nn.init.normal_(w, std=0.01)` |
| `xavier` | `Var[W] = 1/n_vào` theo slide (khi `n_vào = n_ra`); `xavier_normal_` dùng `2/(n_vào+n_ra)`, nêu rõ bạn dùng công thức nào | `nn.init.xavier_normal_` |
| `he` | `Var[W] = 2/n_vào` (baseline) | `nn.init.kaiming_normal_(w, nonlinearity="relu")` |

- **Dự đoán trước khi chạy `zeros`**: điều gì xảy ra với gradient của các nơ-ron trong cùng một lớp? Mạng có học được không? Vì sao? (Từ khoá: đối xứng, ReLU(0).)
- Với mỗi cách khởi tạo, ghi lại **độ lệch chuẩn của kích hoạt sau mỗi lớp ở bước 0** (một lô val) và loss bước 0. Đối chiếu với biểu đồ "30 lớp ReLU" trong slide (Chương 4).
- Mạng 3 lớp có đủ sâu để thấy rõ khác biệt không? Nếu không, nêu nhận xét; hoặc (tuỳ chọn) thử mạng sâu hơn nhiều để thấy hiện tượng như slide.

### Kết hợp và cấu hình cuối cùng (tuỳ chọn)
Nếu bạn muốn, kết hợp các kỹ thuật thấy có ích (chọn **theo val**) thành một cấu hình cuối cùng và chạy với 2–3 seed. Việc chấm điểm trên tập **eval** (cho baseline và cấu hình cuối cùng) làm ở Part 4. Mục *Cải thiện so với baseline của chính bạn* trong RUBRIC xét hai cấu hình này.

**Tự kiểm tra Part 3:**
- [x] Mỗi thí nghiệm đã chạy có ảnh riêng, có dòng trong bảng, có dự đoán trước và đối chiếu sau.
- [x] Mỗi thí nghiệm chỉ đổi một yếu tố (hoặc bạn đã ghi rõ trong `notes`).
- [x] Mọi kết luận "tốt hơn / tệ hơn" được so với độ nhiễu seed (cột "vượt nhiễu?" trong bảng mẫu), hoặc bạn nêu rõ vì sao chưa kết luận được.

---

## Part 4 — Bảng, ảnh, báo cáo, đóng gói

### 4a. Bảng `experiments.xlsx`
Mở [`templates/experiment_table_template.xlsx`](templates/experiment_table_template.xlsx), lưu thành `experiments.xlsx` và điền:
- Sheet `Experiments`: mỗi lần chạy **một dòng**. Ô vàng là ô bạn điền; cột công thức (xám) tự tính. Thêm dòng tuỳ ý. Đừng đổi tên cột.
- Sheet `Seeds`: ghi `exp_id` của các lần chạy baseline với seed khác nhau → tự tính trung bình, độ lệch chuẩn, ngưỡng nhiễu 2σ.
- Sheet `Summary`: tự tổng hợp theo nhóm; bạn viết nhận xét ngắn.
- Cột `eval_acc` / `eval_macro_f1` chỉ điền cho baseline và cấu hình cuối cùng, bằng số do `scripts/evaluate.py` in ra (không tự tính lại bằng cách khác).
- **Tạo bảng bằng code** (khung `code/results_table.py`): ghi `results/*.json` rồi điền vào mẫu bằng `openpyxl`. Nhập tay nhiều dòng dễ sai.

### 4b. Đánh giá cuối trên eval (chỉ làm sau khi đã chọn cấu hình bằng val)
1. Chọn cấu hình cuối cùng chỉ bằng val (có thể chính là baseline nếu bạn không kết hợp thêm gì). Nạp trọng số của **epoch có val loss thấp nhất**.
2. Dự đoán trên **toàn bộ** `X_eval` ở chế độ `eval()` (không dropout), ghi `predictions_eval.csv` với hai cột `row_id,pred` (`pred` ∈ 0..6, đủ 116 203 dòng, mỗi `row_id` đúng một lần).
3. Chạy
   ```bash
   python scripts/evaluate.py --pred submission_<MSSV>/predictions_eval.csv --out submission_<MSSV>/eval_result.json
   ```
   Nếu script báo "FILE DỰ ĐOÁN KHÔNG HỢP LỆ" thì sửa file theo thông báo (thiếu `row_id`, nhãn chưa trừ 1, ...).
4. Ghi `accuracy` và `macro_f1` vào bảng và báo cáo. **Không chỉnh lại cấu hình vì thấy điểm eval chưa như ý** (làm vậy là dùng eval để chọn cấu hình, bị trừ điểm).
5. **Phân tích lỗi:** từ F1 từng lớp và ma trận nhầm lẫn trong `eval_result.json`, nêu lớp khó nhất, nó hay bị nhầm với lớp nào và vì sao.
6. Để có độ nhiễu của điểm eval, bạn có thể chạy cấu hình cuối cùng với vài seed và báo cáo trung bình ± độ lệch chuẩn. File nộp là dự đoán của **một** mô hình (bạn nêu rõ seed nào).

### 4c. Ảnh
Mỗi `exp_id` một ảnh `figures/<exp_id>.png` (xem 2a, khung `code/plots.py`). Mỗi nhóm nên có thêm một ảnh chồng các đường `compare_<nhóm>.png` để so sánh trực tiếp.

### 4d. Báo cáo `REPORT.md`
Dùng [`templates/REPORT_TEMPLATE.md`](templates/REPORT_TEMPLATE.md), tối đa ~4 trang. Chỉ viết cho các chủ đề bạn đã thử. Mỗi kết luận phải có: **con số (trỏ về `exp_id`) + ảnh + giải thích cơ chế**. Ví dụ mức độ mong đợi:

> "Adam (lr = X) đạt val macro-F1 cao hơn SGD+momentum (`opt-sgdm-…`) Y điểm, lớn hơn 2σ_seed = Z, nên khác biệt có ý nghĩa. Đường val-loss của Adam xuống nhanh hơn ở 3 epoch đầu (hình `compare_optimizer.png`), phù hợp với việc Adam chia bước theo độ lớn gradient của từng tham số."

Không chấp nhận: "Adam tốt hơn." (không có số, không so với nhiễu, không có cơ chế).

### 4e. Đóng gói
- Notebook phải **chạy lại được từ đầu đến cuối** (Restart & Run All) trên Colab/Kaggle. Seed cố định. Output còn lại trong file.
- Cấu trúc thư mục nộp đúng như README, mục 6 (danh sách chi tiết từng file). **Mọi code nằm trong `code/`.**

**Tự kiểm tra Part 4:**
- [x] `experiments.xlsx` mở được, không có ô công thức lỗi, các dòng đã điền đủ cột (hoặc ghi lý do thiếu ở `notes`).
- [x] Số ảnh `figures/<exp_id>.png` = số dòng trong bảng.
- [x] Báo cáo trả lời các câu hỏi dẫn dắt cho những chủ đề bạn đã thử.
- [x] `predictions_eval.csv` qua được `scripts/evaluate.py`; `eval_result.json` có trong thư mục nộp; điểm trong bảng và báo cáo khớp `eval_result.json`.
- [x] Điểm eval chỉ xuất hiện ở baseline và cấu hình cuối cùng.
- [x] Không còn `NotImplementedError` trong `code/`.

---

## Chẩn đoán: khi mô hình không học (Chương 5 của slide)

**Trước khi chạy dài, luôn kiểm tra:**
1. Loss bước 0 ≈ `ln C` = ln 7 ≈ 1,946.
2. Quá khớp được 1 lô nhỏ (2–20 mẫu) với mọi chính quy hoá tắt.
3. Mọi tham số có gradient (khác `None`, khác 0).
4. Ghi lại `train loss`, `val loss`, `grad_norm`.

| Triệu chứng | Nguyên nhân thường gặp | Kiểm tra đầu tiên |
|---|---|---|
| Loss phẳng ngay từ bước đầu, ≈ ln 7 | lr quá thấp; nơ-ron chết; gradient không chảy; nhãn sai | Quá khớp một lô nhỏ; in `grad_norm` của từng lớp |
| Loss bước 0 cao hơn ln 7 nhiều | Điểm số lớp cuối quá lớn: khởi tạo sai, thiếu chuẩn hoá | Giảm tỉ lệ `W` lớp cuối, đặt bias = 0; kiểm tra chuẩn hoá đầu vào |
| Loss giảm nhanh rồi dao động | lr quá cao | Giảm lr 3–10 lần |
| Loss là `inf`/`NaN` | lr quá cao; `log(0)`; tràn số FP16 | Clip gradient; tính loss từ logit; kiểm tra `GradScaler` |
| Train thấp, val cao dần | Quá khớp | Thêm dữ liệu/dropout/weight decay/dừng sớm |
| Train và val cùng cao, khoảng cách nhỏ | Chưa khớp (thiếu năng lực/thời gian) | Tăng độ rộng/độ sâu; huấn luyện lâu hơn |

**Một vài lỗi hay gặp ở lab này:**
- Dùng khởi tạo mặc định của `nn.Linear` rồi ghi là "He" trong bảng.
- Đặt softmax trong model rồi lại dùng `F.cross_entropy` (softmax hai lần).
- Quên đổi nhãn `1..7` → `0..6` (gây lỗi index hoặc loss sai).
- Quên chuyển mô hình về `eval()` khi đo val (dropout vẫn bật).
- Gradient bị cộng dồn vì quên `zero_grad`.
- Lấy trung bình/độ lệch chuẩn từ toàn bộ dữ liệu thay vì từ train.
- Đo `grad_norm` *sau* khi clip → luôn ≤ c, không thấy gai.
- So loss CE và MSE trực tiếp với nhau (khác thang đo).
- Kết luận từ một seed duy nhất.

> Mục tiêu không phải con số accuracy cao nhất, mà là **mô hình bạn hiểu và kiểm chứng được, thí nghiệm công bằng, và kết luận có bằng chứng**.

# Lab Day 1 — Xây dựng mạng nơ-ron và thí nghiệm huấn luyện

**Track 4 · Ngày 1 · VinUniversity AICB 2026**
Bài học liên quan: *Mạng Nơ-ron và Huấn Luyện* (slide Day 1).

> Câu hỏi của bài học: *"Một mạng có loss không giảm sau 2 000 bước huấn luyện. Lỗi nằm ở dữ liệu, ở kiến trúc, hay ở vòng lặp huấn luyện?"*
> Sau lab này bạn phải tự trả lời được câu hỏi đó **bằng số liệu do chính bạn đo**.

---

## 1. Bạn sẽ làm gì

Repo này có **dữ liệu đã chia sẵn, script chia/chấm điểm và khung pseudo-code** (`code/`). Bạn **tự hoàn thiện code**; không có đáp án.

1. **Xây dựng model** mạng nơ-ron (MLP) bằng PyTorch theo shape quy định ở mục 3.
2. **Huấn luyện** trên Forest CoverType và theo dõi đường cong huấn luyện.
3. **Thử nghiệm** các yếu tố ảnh hưởng đến huấn luyện: hàm mất mát, bộ tối ưu hoá, hyper-parameter, dropout, gradient clipping, mixed precision, cách khởi tạo tham số. **Bạn tự chọn thử nghiệm nào và bao nhiêu thử nghiệm** (GUIDE, Part 3).
4. **Đánh giá cuối** trên tập `eval` bằng `scripts/evaluate.py` (mục 7).
5. **Tổng hợp**: bảng `.xlsx`, ảnh biểu đồ từng thí nghiệm, báo cáo kết luận, toàn bộ code.

| Phần | Nội dung | Gợi ý |
|---|---|---|
| Part 0 | Chia dữ liệu bằng metadata, tách validation, chuẩn hoá | trên lớp |
| Part 1 | Định nghĩa model, kiểm tra shape và "sức khoẻ" ban đầu | trên lớp |
| Part 2 | Pipeline huấn luyện, baseline, đường cong, độ nhiễu | trên lớp |
| Part 3 | Thí nghiệm (menu gợi ý, tự chọn) | ở nhà |
| Part 4 | Đánh giá cuối trên eval, bảng xlsx, ảnh, báo cáo, đóng gói | ở nhà, nộp trước Day 2 |

## 2. Bài toán và dữ liệu

**Forest CoverType** (Blackard & Dean, UCI): dự đoán loại rừng (7 lớp) từ biến địa hình của các ô đất 30m × 30m ở Colorado.

- 581 012 mẫu, 54 đặc trưng: 10 đặc trưng số liên tục, 4 cột one-hot "Wilderness_Area", 40 cột one-hot "Soil_Type".
- **Mất cân bằng lớp:** lớp 1 (nhãn gốc 2) chiếm 48,8%, lớp 0 chiếm 36,5%, nhãn hiếm nhất (lớp 3) chỉ ~0,5%. Vì thế chỉ số chính là **macro-F1**, kèm accuracy; "đoán luôn lớp đa số" cho accuracy 0,4876 nhưng macro-F1 chỉ ≈ 0,094.
- **Đã có sẵn trong repo, không cần tải:** `data/covtype.csv.gz` (dữ liệu đầy đủ) và `data/split_metadata.csv` (mỗi dòng gán một mẫu cho `train` hoặc `eval`). Mô tả chi tiết cột, nhãn ở [`data/README.md`](data/README.md).

**Chia dữ liệu (cố định cho mọi sinh viên):**

| Tập | Số mẫu | Dùng để |
|---|---|---|
| `train` | 464 809 | huấn luyện; bạn tự tách một phần làm **validation** để chọn cấu hình |
| `eval` | 116 203 | chỉ để chấm điểm cuối, không dùng để chọn gì |

Bạn **không** tự chia lại train/eval và **không** sửa `split_metadata.csv`. Tạo `train.npz` và `eval.npz` bằng:

```bash
python scripts/split_data.py
```

## 3. Kiến trúc mô hình (cố định)

Đầu vào `x`: `(B, 54)` `float32`. Nhãn `y`: `(B,)` `int64` (`0..6`). Đầu ra: **logits `(B, 7)`** (chưa softmax).

| Tên | Dùng ở | Các lớp | Số tham số |
|---|---|---|---|
| `M-base` | **Bắt buộc**: baseline | `54 → 256 → 128 → 7` | 47 879 |
| `M-wide` | Tuỳ chọn: thí nghiệm độ rộng | `54 → 512 → 256 → 7` | 161 287 |
| `M-deep` | Tuỳ chọn: thí nghiệm độ sâu | `54 → 256 → 128 → 64 → 7` | 55 687 |

ReLU ở mọi lớp ẩn, có bias, dropout (nếu dùng) chỉ đặt sau ReLU của lớp ẩn, không BatchNorm hay residual. Chi tiết trong [`GUIDE.md`](GUIDE.md), mục *Quy định kiến trúc mô hình*.

## 4. Chạy ở đâu

Khuyến nghị dùng GPU miễn phí:

- **Google Colab:** *Runtime → Change runtime type → T4 GPU*. Nên mount Google Drive để lưu kết quả (Colab có thể ngắt kết nối giữa chừng).
- **Kaggle Notebook:** *Settings → Accelerator → GPU*. Nếu clone repo từ GitHub cần bật Internet.
- CPU vẫn chạy được nhưng chậm hơn nhiều khi thử nhiều cấu hình.

Phần mềm: Python 3.10+, PyTorch ≥ 2.x, scikit-learn, numpy, pandas, matplotlib, openpyxl. Colab/Kaggle đã cài sẵn.

## 5. Luật chơi

**Bạn phải tự viết:** model (class `nn.Module` do bạn định nghĩa, đúng shape ở mục 3), pipeline huấn luyện và đánh giá, ghi log, vẽ biểu đồ, lập bảng. Khung `code/` chỉ có chữ ký hàm và các bước gợi ý.

**Được dùng thoải mái:** mọi thành phần của PyTorch: `nn.Linear`, `nn.ReLU`, `nn.Dropout`, `nn.Sequential`, `nn.init.*`, `nn.CrossEntropyLoss`, `nn.MSELoss`, `torch.optim.*`, autograd, `clip_grad_norm_`, `torch.autocast`, `GradScaler`; cùng `numpy`, `sklearn` (tách validation, tính metric), `matplotlib`.

**Không được dùng:** mô hình dựng sẵn hoặc pretrained (`torchvision`/`timm`), `sklearn.neural_network.MLPClassifier`, hay copy mạng làm sẵn thay cho model do bạn định nghĩa.

**Công bằng khi so sánh:**
1. Mọi thí nghiệm dùng **cùng phép tách validation** và **cùng số epoch** (khuyến nghị 20), trừ yếu tố bạn đang thử.
2. **Mỗi lần chỉ đổi một yếu tố** so với baseline (hoặc ghi rõ khi bạn đổi nhiều yếu tố).
3. **Không dùng tập eval để chọn cấu hình** (lr, epoch, kiến trúc, kỹ thuật, cách chuẩn hoá). Chọn bằng validation. Eval chỉ dùng ở cuối, cho baseline và cấu hình cuối cùng.
4. Mọi con số trong báo cáo phải truy được về một dòng trong bảng `.xlsx`. Mọi kết luận "A tốt hơn B" phải cân nhắc **độ nhiễu giữa các seed** (chạy baseline vài seed để đo), hoặc nêu rõ đây là hạn chế.

## 6. Sản phẩm phải nộp

Nộp **một thư mục `submission_<MSSV>/`** (thay `<MSSV>` bằng mã số sinh viên của bạn), nén thành `submission_<MSSV>.zip` nếu nộp qua LMS, theo kênh giảng viên thông báo.

### 6.1 Cây thư mục

```
submission_<MSSV>/
├── REPORT.md
├── experiments.xlsx
├── predictions_eval.csv
├── eval_result.json
├── figures/
│   ├── <exp_id>.png            (một ảnh cho MỖI thí nghiệm trong bảng)
│   └── compare_<nhóm>.png      (nên có, mỗi nhóm thí nghiệm một ảnh)
├── results/                    (khuyến nghị, không bắt buộc)
│   └── <exp_id>.json
└── code/                       (TOÀN BỘ code nằm ở đây)
    ├── lab.ipynb
    ├── data.py  model.py  optimizer.py  train.py  plots.py  results_table.py
    └── requirements.txt        (tuỳ chọn)
```

### 6.2 Chi tiết từng file / thư mục

| Đường dẫn | Bắt buộc? | Nội dung và yêu cầu |
|---|---|---|
| `REPORT.md` | **Bắt buộc** | Báo cáo kết luận, viết theo [`templates/REPORT_TEMPLATE.md`](templates/REPORT_TEMPLATE.md), khoảng 4 trang. Gồm: thiết lập, kiểm tra ban đầu và độ nhiễu, kết quả theo từng chủ đề đã thử (dự đoán, số liệu trỏ về `exp_id`, ảnh, giải thích cơ chế), **đánh giá cuối trên eval và phân tích lỗi theo lớp**, trả lời câu hỏi dẫn dắt, hạn chế. Chèn ảnh bằng đường dẫn tương đối, ví dụ `![](figures/compare_optimizer.png)`. Chỉ viết cho các chủ đề bạn đã thử. |
| `experiments.xlsx` | **Bắt buộc** | Bảng so sánh, tạo từ [`templates/experiment_table_template.xlsx`](templates/experiment_table_template.xlsx). Giữ nguyên 4 sheet `Legend`, `Experiments`, `Seeds`, `Summary` và không đổi tên cột. **Mỗi thí nghiệm đã chạy là một dòng** trong sheet `Experiments`, với `exp_id` duy nhất. Điền đủ cấu hình và kết quả (hoặc ghi lý do thiếu vào `notes`). Cột `eval_acc` / `eval_macro_f1` chỉ điền cho baseline và cấu hình cuối cùng, bằng số do `evaluate.py` in ra. Sheet `Seeds` ghi các lần chạy baseline khác seed. Không để ô công thức lỗi. |
| `predictions_eval.csv` | **Bắt buộc** | Dự đoán của **cấu hình cuối cùng** trên toàn bộ tập eval. CSV có tiêu đề `row_id,pred`; `row_id` lấy từ `eval.npz`, `pred` là số nguyên 0..6 (cùng hệ nhãn với `y`, tức đã trừ 1); đủ 116 203 dòng, mỗi `row_id` đúng một lần. Phải qua được `scripts/evaluate.py`. |
| `eval_result.json` | **Bắt buộc** | Đầu ra của `python scripts/evaluate.py --pred predictions_eval.csv --out eval_result.json` (accuracy, macro-F1, F1 từng lớp, ma trận nhầm lẫn). Không tự sửa. Số trong báo cáo và bảng phải khớp file này. |
| `figures/` | **Bắt buộc** | Thư mục chứa ảnh biểu đồ, định dạng `.png`. |
| `figures/<exp_id>.png` | **Bắt buộc, một ảnh cho mỗi dòng của bảng** | Tên file **trùng đúng** `exp_id` trong bảng (ví dụ `base-s1.png`, `opt-adam-lr1e-3.png`). Mỗi ảnh có ít nhất 3 ô: (1) train loss và val loss theo epoch, (2) val accuracy (nên có thêm macro-F1), (3) `grad_norm` (đo trước khi clip). Có tiêu đề ghi `exp_id` và cấu hình, nhãn trục, chú thích. Là ảnh lưu từ `plt.savefig` hoặc ảnh chụp màn hình TensorBoard/W&B đều được. Tên file ghi vào cột `figure_file` của bảng. Số ảnh phải bằng số dòng của bảng. |
| `figures/compare_<nhóm>.png` | Nên có | Ảnh chồng các đường của nhiều thí nghiệm cùng một nhóm (ví dụ `compare_optimizer.png`, `compare_dropout.png`) để so sánh trực tiếp. Dùng làm bằng chứng trong báo cáo. |
| `results/` và `results/<exp_id>.json` | Khuyến nghị | Lịch sử của từng lần chạy (mỗi epoch: train/val loss, val acc, macro-F1, grad_norm, thời gian; cùng cấu hình `cfg`). Giúp tạo lại bảng và ảnh mà không cần huấn luyện lại. Một file cho mỗi `exp_id`. |
| `code/` | **Bắt buộc** | **Toàn bộ code của bạn nằm trong thư mục này**, không để file code ở nơi khác. Bắt đầu từ khung `code/` của repo (đã hoàn thiện, không còn `NotImplementedError`). |
| `code/lab.ipynb` | **Bắt buộc** | Notebook chạy được từ đầu đến cuối (*Restart & Run All*) trên Colab hoặc Kaggle, **giữ nguyên output** của các ô. Theo khung các mục Part 0–4; mỗi thí nghiệm có dự đoán trước và nhận xét sau. Đường dẫn ghi kết quả đúng khi chạy từ `submission_<MSSV>/code/`: ghi vào `../figures/`, `../results/`; dữ liệu ở `../../data`. Đặt seed cố định. |
| `code/data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py` | **Bắt buộc** (bản hoàn thiện của khung) | Các module mà notebook import. Bạn có thể đổi chữ ký hàm, thêm file, miễn là notebook chạy được khi chỉ có thư mục nộp cùng `data/` và `scripts/` của repo. |
| `code/requirements.txt` | Tuỳ chọn | Liệt kê thư viện và phiên bản nếu bạn dùng thư viện ngoài những cái có sẵn trên Colab/Kaggle. |

### 6.3 Không nộp

- Dữ liệu (`data/`, `*.npz`, `covtype.csv.gz`) và thư mục cache của sklearn: chúng đã có trong repo.
- File trọng số mô hình (`.pt`, `.pth`, `.ckpt`).
- `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`.
- Bản sao slide, file PDF, hay ảnh nằm ngoài `figures/`.
- Code đặt ngoài thư mục `code/`.

### 6.4 Kiểm tra nhanh trước khi nộp

- [x] Tên thư mục là `submission_<MSSV>` và có đủ `REPORT.md`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, `figures/`, `code/lab.ipynb`.
- [x] `python scripts/evaluate.py --pred submission_<MSSV>/predictions_eval.csv` chạy thành công; số khớp `eval_result.json`, bảng và báo cáo.
- [x] Số ảnh `figures/<exp_id>.png` bằng số dòng thí nghiệm trong `experiments.xlsx`, và tên ảnh trùng `exp_id`.
- [x] Mọi file `.py` và notebook đều nằm trong `code/`; không còn `NotImplementedError`.
- [x] Mở `code/lab.ipynb` trên Colab/Kaggle, chọn *Restart & Run All* không lỗi, output còn nguyên.
- [x] Mọi con số trong `REPORT.md` tìm lại được trong `experiments.xlsx`.

## 7. Cách đánh giá

- **Chỉ số chính:** macro-F1 trên tập `eval` (trung bình cộng F1 của 7 lớp, mỗi lớp trọng số như nhau). **Chỉ số phụ:** accuracy.
- **Cách chấm:** bạn nộp `predictions_eval.csv`; giảng viên chạy đúng `scripts/evaluate.py` để tính điểm. Bạn nên chạy thử script này trước khi nộp.
- **Chọn cấu hình bằng val, không bằng eval.** Mỗi lần chạy `evaluate.py` chỉ dành cho baseline và cấu hình cuối cùng.
- Điểm eval chiếm **15/100**: file hợp lệ và khớp số (4), mức macro-F1 (5), cải thiện so với baseline của chính bạn (3), phân tích lỗi theo lớp (3). Mức macro-F1 chi tiết ở [`RUBRIC.md`](RUBRIC.md) mục 7. Quy trình và công thức ở [`GUIDE.md`](GUIDE.md), mục *Cách đánh giá*.

## 8. Quy trình làm bài

```bash
git pull                                              # lấy repo (hoặc: git clone <url của repo>)
python scripts/split_data.py                          # tạo data/processed/train.npz và eval.npz
mkdir -p submission_<MSSV> && cp -r code submission_<MSSV>/code
# mở submission_<MSSV>/code/lab.ipynb, hoàn thiện các file .py và notebook
python scripts/evaluate.py --pred submission_<MSSV>/predictions_eval.csv --out submission_<MSSV>/eval_result.json
```

Trên Colab: `git clone` repo vào `/content`, rồi làm tương tự (xem ô đầu của `lab.ipynb`).

## 9. Các file trong repo

| Đường dẫn | Dùng để làm gì |
|---|---|
| [`GUIDE.md`](GUIDE.md) | Hướng dẫn từng bước, cách đánh giá, menu thí nghiệm, checklist tự kiểm tra |
| [`RUBRIC.md`](RUBRIC.md) | Thang điểm chi tiết (đọc trước khi bắt đầu) |
| [`data/`](data/README.md) | Dữ liệu CoverType và `split_metadata.csv` (chia train/eval) |
| `scripts/split_data.py` | Chia dữ liệu theo metadata → `data/processed/*.npz` |
| `scripts/evaluate.py` | Chấm `predictions_eval.csv` (accuracy, macro-F1, F1 từng lớp, ma trận nhầm lẫn) |
| `scripts/make_split_metadata.py` | (giảng viên) tạo lại metadata; sinh viên không cần chạy |
| `code/` | Khung pseudo-code: `data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`, `lab.ipynb` |
| [`templates/experiment_table_template.xlsx`](templates/experiment_table_template.xlsx) | Bảng so sánh có sẵn tên cột và công thức |
| [`templates/REPORT_TEMPLATE.md`](templates/REPORT_TEMPLATE.md) | Khung báo cáo |

## 10. Tài liệu tham khảo

Slide Day 1 (đặc biệt Chương 3, 4, 5); Karpathy, *A Recipe for Training Neural Networks*; CS231n, *Neural Networks Part 1–3*; Kingma & Ba (2015) *Adam*; Loshchilov & Hutter (2019) *AdamW*; He et al. (2015) *khởi tạo He*; Srivastava et al. (2014) *Dropout*; Micikevicius et al. (2018) *Mixed Precision Training*; Blackard & Dean (1999) về dataset CoverType.

Nếu kẹt, hãy bắt đầu với **mục "Chẩn đoán" ở cuối GUIDE**.

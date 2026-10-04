# RUBRIC — Lab Day 1 (100 điểm)

Điểm đánh giá **chất lượng thí nghiệm, lập luận và điểm trên tập eval**. Điểm eval chiếm 15/100; phần còn lại chấm quy trình, nên một mô hình điểm cao nhưng chọn cấu hình bằng cách nhìn eval sẽ bị trừ điểm.

**Số lượng thí nghiệm không bị ép.** Bạn tự chọn chạy bao nhiêu thí nghiệm. Điểm phần thí nghiệm dựa vào *độ phủ chủ đề* (tối đa 7 chủ đề) và *chất lượng thiết kế*, không dựa vào việc chạy đủ một danh sách cố định.

| # | Phần | Điểm |
|---|---|---|
| 1 | Dữ liệu, pipeline và kiểm tra ban đầu | 8 |
| 2 | Model và baseline | 10 |
| 3 | Độ phủ chủ đề thí nghiệm | 14 |
| 4 | Chất lượng thiết kế thí nghiệm | 8 |
| 5 | Báo cáo kết luận | 20 |
| 6 | Bảng `experiments.xlsx` và ảnh biểu đồ | 15 |
| 7 | **Điểm đánh giá trên tập eval** | 15 |
| 8 | Chất lượng code và nộp bài | 10 |
| | **Tổng** | **100** |

---

## 1. Dữ liệu, pipeline và kiểm tra ban đầu (8)

| Tiêu chí | Điểm |
|---|---|
| Chạy `scripts/split_data.py`, tách validation **từ train** (phân tầng, seed cố định), chuẩn hoá chỉ bằng thống kê train, không đưa eval vào bất kỳ bước nào trước đánh giá cuối | 3 |
| `run_experiment(cfg, data)` ghi đủ: train/val loss (train đo ở eval mode), val acc, val macro-F1, `grad_norm` **trước clip**, loss bước 0, thời gian, bộ nhớ, cờ diverged; cấu hình tách khỏi logic | 3 |
| Hai phép thử sức khoẻ: loss bước 0 ≈ ln 7 (có in và nhận xét); quá khớp được 20 mẫu (có đường cong) | 2 |

## 2. Model và baseline (10)

| Tiêu chí | Điểm |
|---|---|
| Model **tự định nghĩa** đúng shape quy định (`M-base`: logits `(B,7)`, **47 879 tham số**, có `assert`), dropout đúng chỗ (sau ReLU lớp ẩn), softmax không nằm trong model | 3 |
| Khởi tạo đúng với cái ghi trong bảng (không để mặc định `nn.Linear` mà ghi là He); có kiểm tra gradient chảy tới mọi tham số | 2 |
| Baseline đúng cấu hình, đường cong train/val bình thường, vượt mốc "đoán đa số" (accuracy 0,4876), có mô tả hình dạng đường cong | 2 |
| Đo độ nhiễu seed (≥ 2 seed baseline, trung bình ± độ lệch chuẩn); nếu chỉ 1 seed phải nêu rõ là hạn chế (tối đa 1/3 điểm mục này) | 3 |

## 3. Độ phủ chủ đề thí nghiệm (14)

7 chủ đề, **mỗi chủ đề tối đa 2 điểm**: hàm mất mát · bộ tối ưu hoá · hyper-parameter · dropout · gradient clipping · mixed precision · khởi tạo tham số.

Một chủ đề được tính đủ 2 điểm khi: có ≥ 1 thí nghiệm chạy xong và so với baseline, có dòng trong bảng, có ảnh riêng, có dự đoán trước và đối chiếu sau. Thí nghiệm chạy nhưng thiếu ảnh/bảng/đối chiếu: 1 điểm. Không có thí nghiệm: 0.

Bạn không cần phủ hết 7 chủ đề để đạt điểm cao ở các phần khác, nhưng mỗi chủ đề bỏ qua là mất 2 điểm ở phần này.

## 4. Chất lượng thiết kế thí nghiệm (8)

Chấm trên các thí nghiệm bạn đã chạy:

| Tiêu chí | Điểm |
|---|---|
| **Công bằng:** mỗi lần chỉ đổi một yếu tố; cùng split, cùng số epoch, cùng seed (hoặc ghi rõ); nếu so bộ tối ưu thì mỗi bộ được thử ≥ 2 lr và so ở lr tốt nhất của nó | 3 |
| **Chọn tham số có lý do:** `c` của clipping chọn dựa trên `grad_norm`; kiểm tra clipping ở lr cao; `q` của dropout có liên hệ với mức quá khớp; so CE/MSE bằng metric chứ không bằng loss | 2 |
| **Dự đoán trước, đối chiếu sau** cho các thí nghiệm | 2 |
| **Chọn cấu hình chỉ bằng val** (xem mục trừ điểm) | 1 |

## 5. Báo cáo kết luận (20)

| Tiêu chí | Điểm |
|---|---|
| Mọi kết luận có số liệu (trỏ về `exp_id`), ảnh và **giải thích cơ chế** (vì sao xảy ra: gradient, phương sai, số bước cập nhật, ...) | 8 |
| Cân nhắc độ nhiễu seed khi nói "A tốt hơn B"; chênh lệch nhỏ hơn nhiễu được nêu là chưa kết luận được | 4 |
| Trả lời các câu hỏi dẫn dắt trong mẫu cho những chủ đề đã thử, trong đó có câu hỏi quay lại bài học (3 phép kiểm tra đầu tiên khi loss không giảm) | 4 |
| Trung thực: nêu hạn chế, kết quả "không như dự đoán" và giải thích; tách rõ điều đã đo với điều chỉ phỏng đoán | 4 |

## 6. Bảng `experiments.xlsx` và ảnh biểu đồ (15)

| Tiêu chí | Điểm |
|---|---|
| Bảng đầy đủ cột, mỗi thí nghiệm một dòng, giá trị khớp với notebook và báo cáo, không ô công thức lỗi | 6 |
| **Mỗi thí nghiệm có ảnh riêng** `figures/<exp_id>.png` (train/val loss, val acc, grad_norm), đặt đúng tên, đọc được (trục, chú thích, tiêu đề) | 6 |
| Ảnh chồng so sánh theo nhóm (`compare_<nhóm>.png`) và sheet Summary có nhận xét | 3 |

## 7. Điểm đánh giá trên tập eval (15)

Chấm bằng đúng `scripts/evaluate.py` trên `predictions_eval.csv` của bạn. **Chỉ số chính: macro-F1.** Accuracy chỉ để tham khảo và báo cáo.

| Tiêu chí | Điểm | Cách chấm |
|---|---|---|
| **File hợp lệ và khớp** | 4 | `predictions_eval.csv` qua `evaluate.py` (đủ 116 203 `row_id`, nhãn 0..6); macro-F1 và accuracy trong báo cáo/bảng khớp `eval_result.json` (sai lệch ≤ 0,0005); có `eval_result.json` |
| **Mức macro-F1 trên eval** | 5 | theo bảng bên dưới |
| **Cải thiện so với baseline của chính bạn** | 3 | macro-F1 eval của cấu hình cuối cùng so với baseline: cải thiện ≥ 0,02 và vượt nhiễu seed → 3; từ 0,005 đến dưới 0,02 → 2; từ 0 đến dưới 0,005 → 1; thấp hơn baseline → 0. Nếu bạn nộp chính baseline làm cấu hình cuối cùng thì ô này tối đa 1 |
| **Phân tích lỗi theo lớp** | 3 | có bảng precision/recall/F1 từng lớp và ma trận nhầm lẫn; nêu lớp khó nhất, nó bị nhầm với lớp nào và lý giải bằng dữ liệu (số mẫu, đặc trưng) |

**Mức macro-F1 trên eval (5 điểm):**

| macro-F1 (eval) | Điểm |
|---|---|
| < 0,60 | 0 |
| 0,60 đến < 0,75 | 1 |
| 0,75 đến < 0,80 | 2 |
| 0,80 đến < 0,83 | 3 |
| 0,83 đến < 0,86 | 4 |
| ≥ 0,86 | 5 |

Các mức được hiệu chỉnh từ vài lần chạy tham chiếu của giảng viên (1 seed, một máy, trên đúng phép chia này): `M-base` 20 epoch cho macro-F1 trên eval khoảng 0,83 đến 0,87 tuỳ bộ tối ưu và lr; chạy rất ngắn với lr quá nhỏ chỉ khoảng 0,57; đoán luôn lớp đa số khoảng 0,09. Mức ≥ 0,86 đòi hỏi chọn lr hợp lý (hoặc huấn luyện đủ lâu).

**Điều kiện:** nếu có bằng chứng dùng eval để chọn cấu hình (xem mục trừ điểm), phần "Mức macro-F1" và "Cải thiện" bị tính 0.

## 8. Chất lượng code và nộp bài (10)

| Tiêu chí | Điểm |
|---|---|
| Notebook chạy lại được từ đầu đến cuối (Restart & Run All), seed cố định, có output | 4 |
| Hoàn thiện khung `code/`: không còn `NotImplementedError`, model tách riêng, `run_experiment` duy nhất, chú thích chỗ khó | 3 |
| Nộp đúng cấu trúc README mục 6: đủ `REPORT.md`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, `figures/`, `code/lab.ipynb`; **toàn bộ code nằm trong `code/`**; tên ảnh trùng `exp_id`; không nộp dữ liệu/checkpoint | 3 |

---

## Trừ điểm

| Vi phạm | Trừ |
|---|---|
| Không tự định nghĩa model (dùng mô hình dựng sẵn/pretrained/`MLPClassifier`) | Phần 2 = 0 |
| Dùng tập eval để chọn cấu hình/lr/epoch/chuẩn hoá, hoặc chỉnh lại sau khi thấy điểm eval | −10 và mục 7 (hai ô điểm) = 0 |
| Sửa `split_metadata.csv` hoặc tự chia lại train/eval | −10 |
| Kết luận so sánh nhưng không nhắc đến nhiễu seed hay hạn chế | −1 đến −3 mỗi chỗ |
| Số trong báo cáo không khớp bảng/ảnh/`eval_result.json` | −1 đến −5 |
| Shape hoặc số tham số model sai quy định mà không giải thích | −2 |
| Nộp trễ | Theo quy định của khoá học |

## Bậc xếp loại gợi ý

| Điểm | Mức |
|---|---|
| ≥ 90 | Xuất sắc: so sánh công bằng, giải thích được cơ chế, phủ rộng chủ đề, điểm eval tốt |
| 75–89 | Tốt: thí nghiệm đủ ý, kết luận có số liệu, còn vài chỗ thiếu cơ chế/nhiễu |
| 60–74 | Đạt: chạy được, nhưng so sánh chưa công bằng hoặc kết luận chung chung |
| < 60 | Chưa đạt: thiếu phần lớn thí nghiệm/bảng/ảnh hoặc model sai quy định |

## Checklist tự chấm trước khi nộp

- [x] Model đúng shape, `assert` 47 879 tham số
- [x] Loss bước 0 ≈ 1,946 và quá khớp được 20 mẫu
- [x] Mỗi thí nghiệm: 1 dòng bảng + 1 ảnh + dự đoán trước + đối chiếu sau
- [x] Số ảnh = số dòng trong bảng
- [x] Chọn cấu hình chỉ bằng val; chạy `scripts/evaluate.py` đúng một lần cho cấu hình cuối cùng
- [x] `predictions_eval.csv` hợp lệ; `eval_result.json` khớp bảng và báo cáo
- [x] Có phân tích lỗi theo lớp (bảng F1 từng lớp + ma trận nhầm lẫn)
- [x] Báo cáo: mỗi câu "A hơn B" có số, ảnh, cơ chế và nhắc nhiễu seed
- [x] Notebook chạy lại từ đầu không lỗi; không còn `NotImplementedError`
- [x] Đủ file theo README mục 6.2; code nằm hết trong `code/`; không có dữ liệu, `.pt`, `__pycache__`

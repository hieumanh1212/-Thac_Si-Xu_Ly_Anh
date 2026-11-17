# Nhận dạng biển số xe máy – Ứng dụng Web

Bài tập lớn môn **Xử lý ảnh**. Hệ thống dùng các kỹ thuật xử lý ảnh cổ điển (OpenCV) để định vị
và phân đoạn biển số, sau đó nhận dạng ký tự bằng EasyOCR. Giao diện web hiển thị **từng bước xử lý
trung gian**, giúp minh hoạ thuật toán khi báo cáo.

## Cài đặt (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # PyTorch bản CPU (nhẹ hơn)
pip install -r requirements.txt
```

Lần chạy đầu EasyOCR sẽ tự tải mô hình (~100 MB) về `~/.EasyOCR/`.

## Chạy

```powershell
python app.py
```

Mở trình duyệt: <http://127.0.0.1:5000>. Bạn có thể tải ảnh lên, kéo thả, dán (Ctrl+V) hoặc chụp từ webcam.

## Kiểm thử và đánh giá

```powershell
python tools/make_samples.py --out samples --n 20     # sinh ảnh tổng hợp có nhãn
python tools/evaluate.py samples --csv ket_qua.csv     # đo độ chính xác
```

Để đánh giá trên ảnh thật, đặt tên file theo nhãn: `59X1_12345.jpg`, `29B1_1234.jpg`, ...

## Pipeline xử lý

| Bước | Kỹ thuật | Module |
|---|---|---|
| 1. Tiền xử lý | Ảnh xám → CLAHE → lọc song phương → Canny → phép đóng; ngưỡng Otsu + phép mở tìm vùng sáng | `lpr/preprocess.py` |
| 2. Định vị biển số | Tìm contour, lọc theo diện tích, độ đầy, tỉ lệ rộng/cao (~1.36), xấp xỉ đa giác 4 đỉnh | `lpr/detect.py` |
| 3. Nắn phối cảnh | Perspective transform về 380×280 | `lpr/utils.py` |
| 4. Nhị phân hoá và phân đoạn | Adaptive threshold → connected components → lọc kích thước → tách 2 dòng theo y | `lpr/segment.py` |
| 5. Chọn ứng viên | Chấm điểm mỗi ứng viên theo cấu trúc ký tự (2 dòng, 3–5 + 4–5 ký tự) | `lpr/segment.py` |
| 6. Nhận dạng ký tự | EasyOCR (CRNN) trên từng dòng; dòng 2 chỉ cho phép chữ số | `lpr/recognize.py` |
| 7. Hậu xử lý | Sửa nhầm lẫn chữ/số theo vị trí (O→0, B→8, ...), định dạng `59-X1 123.45` | `lpr/postprocess.py` |

## Cấu trúc thư mục

```
app.py                 Flask backend (API /api/recognize, /api/status)
lpr/                   Thư viện xử lý ảnh
templates/index.html   Giao diện
static/                CSS, JavaScript
tools/                 Sinh ảnh mẫu, đánh giá
```

## Hạn chế và hướng phát triển

- Phương pháp contour nhạy với ảnh có nhiều cạnh nhiễu, biển số quá nhỏ, bị che hoặc nghiêng mạnh.
- Có thể thay bước định vị bằng YOLO, và thay EasyOCR bằng bộ phân loại ký tự tự huấn luyện (KNN/SVM/CNN).
- Bộ dữ liệu ảnh biển số Việt Nam công khai có thể tìm trên GitHub/Kaggle với từ khoá
  "Vietnamese license plate dataset".

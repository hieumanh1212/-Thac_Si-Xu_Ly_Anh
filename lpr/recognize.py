"""Bước 4 - Nhận dạng ký tự bằng EasyOCR (mạng CRNN: CNN + LSTM + CTC).

Ta KHÔNG dùng bộ phát hiện chữ của EasyOCR mà tự cắt từng dòng ký tự (kết quả của bước phân đoạn),
rồi gọi `reader.recognize()` để chỉ chạy phần nhận dạng trên dòng đó. Mỗi dòng có tập ký tự cho phép
riêng (dòng 2 chỉ gồm chữ số) giúp giảm nhầm lẫn đáng kể.
"""

import threading

import cv2
import numpy as np

from .postprocess import DIGITS, LETTERS


class OCREngine:
    def __init__(self, langs=("en",), gpu=False):
        self.langs = list(langs)
        self.gpu = gpu
        self._reader = None
        self._lock = threading.Lock()
        self.error = None
        self.loading = False

    @property
    def ready(self):
        return self._reader is not None

    def load(self):
        if self._reader is not None or self.error:
            return self._reader
        with self._lock:
            if self._reader is None and not self.error:
                self.loading = True
                try:
                    import easyocr  # import trễ vì nặng (PyTorch)
                    self._reader = easyocr.Reader(self.langs, gpu=self.gpu, verbose=False)
                except Exception as exc:  # noqa: BLE001
                    self.error = f"Không khởi tạo được EasyOCR: {exc}"
                finally:
                    self.loading = False
        return self._reader

    @staticmethod
    def _prepare(gray_line, target_h=64):
        """Thêm viền trắng và phóng to dòng ký tự về chiều cao chuẩn."""
        h, w = gray_line.shape[:2]
        if h == 0 or w == 0:
            return None
        scale = target_h / float(h)
        img = cv2.resize(gray_line, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_CUBIC)
        pad = int(target_h * 0.25)
        bg = int(np.percentile(img, 90))
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=bg)

    def read_line(self, gray_line, allowlist):
        """Trả về (chuỗi, độ tin cậy)."""
        reader = self.load()
        if reader is None:
            return "", 0.0
        img = self._prepare(gray_line)
        if img is None:
            return "", 0.0
        res = reader.recognize(img, allowlist=allowlist, detail=1, paragraph=False)
        if not res:
            return "", 0.0
        res = sorted(res, key=lambda r: r[0][0][0])
        text = "".join(r[1] for r in res)
        conf = float(np.mean([r[2] for r in res]))
        return text, conf


LINE1_ALLOW = DIGITS + LETTERS
LINE2_ALLOW = DIGITS


def line_regions(rows, plate_shape, pad=6):
    """Xác định vùng (y0, y1, x0, x1) của dòng trên và dòng dưới trên biển số."""
    H, W = plate_shape[:2]
    if len(rows) == 2:
        regions = []
        for row in rows:
            x0 = max(0, min(b[0] for b in row) - pad)
            x1 = min(W, max(b[0] + b[2] for b in row) + pad)
            y0 = max(0, min(b[1] for b in row) - pad)
            y1 = min(H, max(b[1] + b[3] for b in row) + pad)
            regions.append((y0, y1, x0, x1))
        return regions
    # Không tách được 2 dòng -> chia đôi biển số theo chiều cao
    m = H // 2
    return [(int(H * 0.05), m, 0, W), (m, int(H * 0.95), 0, W)]

"""Bước 4 (học sâu) - Nhận dạng từng ký tự bằng mạng CNN nhỏ, train riêng cho font biển số VN.

EasyOCR là mô hình đa dụng nên hay nhầm với font chữ hẹp của biển số Việt Nam (V↔1, 7↔1, M→D...).
Ở đây mỗi ký tự đã tách ở bước phân đoạn được phân loại vào 1 trong 36 lớp (0-9, A-Z), có ràng buộc
theo VỊ TRÍ: 2 ký tự đầu dòng trên là chữ số, ký tự thứ 3 là chữ cái, dòng dưới toàn chữ số.

Kiến trúc: 3 khối [Conv3x3-BN-ReLU ×2 → MaxPool] → Dropout → FC(256) → FC(36), đầu vào ảnh xám 32×32.
"""

import threading
from pathlib import Path

import cv2
import numpy as np

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SIZE = 32
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "char_cnn.pt"

DIGITS_IDX = [CLASSES.index(c) for c in "0123456789"]
LETTERS_IDX = [CLASSES.index(c) for c in "ABCDEFGHKLMNPRSTUVXYZ"]  # chữ cái dùng trên biển số VN
ALNUM_IDX = DIGITS_IDX + LETTERS_IDX


def char_to_input(gray_crop):
    """Chuẩn hoá ảnh 1 ký tự: đặt giữa khung vuông nền sáng, resize 32×32, kéo giãn tương phản về [0, 1]."""
    g = gray_crop
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    side = int(max(h, w) * 1.15) + 2
    canvas = np.full((side, side), int(np.percentile(g, 90)), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    canvas[oy:oy + h, ox:ox + w] = g
    canvas = cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    lo, hi = np.percentile(canvas, 2), np.percentile(canvas, 98)
    return np.clip((canvas - lo) / max(hi - lo, 1.0), 0, 1).astype(np.float32)


def build_model():
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                             nn.MaxPool2d(2))

    return nn.Sequential(
        block(1, 32), block(32, 64), block(64, 128),  # 32 → 16 → 8 → 4
        nn.Flatten(), nn.Dropout(0.4), nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
        nn.Dropout(0.3), nn.Linear(256, len(CLASSES)),
    )


def allowed_classes(line, pos):
    """Tập lớp hợp lệ theo vị trí. Dòng trên: [số][số][chữ][số|chữ]..., dòng dưới: toàn số."""
    if line == 1 or pos in (0, 1):
        return DIGITS_IDX
    if pos == 2:
        return LETTERS_IDX
    return ALNUM_IDX


class CharClassifier:
    def __init__(self, weights=MODEL_PATH):
        self.weights = Path(weights)
        self._model = None
        self._lock = threading.Lock()
        self.error = None

    @property
    def available(self):
        return self.weights.exists() and self.error is None

    def load(self):
        if self._model is not None or not self.available:
            return self._model
        with self._lock:
            if self._model is None and self.error is None:
                try:
                    import torch
                    model = build_model()
                    model.load_state_dict(torch.load(str(self.weights), map_location="cpu", weights_only=True))
                    model.eval()
                    self._model = model
                except Exception as exc:  # noqa: BLE001
                    self.error = f"Không nạp được CNN ký tự: {exc}"
        return self._model

    def read_rows(self, gray_plate, rows, pad=2):
        """Phân loại các ký tự của biển 2 dòng. Trả về ([dòng 1, dòng 2], độ tin cậy) hoặc (None, 0)."""
        model = self.load()
        if model is None:
            return None, 0.0
        import torch

        H, W = gray_plate.shape[:2]
        crops, meta = [], []
        for li, row in enumerate(rows):
            for pi, (x, y, w, h) in enumerate(row):
                crop = gray_plate[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
                crops.append(char_to_input(crop))
                meta.append((li, pi))
        with torch.no_grad():
            probs = torch.softmax(model(torch.from_numpy(np.stack(crops)[:, None])), dim=1).numpy()

        texts, confs = ["", ""], []
        for (li, pi), p in zip(meta, probs):
            allowed = allowed_classes(li, pi)
            k = allowed[int(np.argmax(p[allowed]))]
            texts[li] += CLASSES[k]
            confs.append(float(p[k]))
        # Độ tin cậy của biển = độ tin cậy của ký tự KÉM NHẤT (1 ký tự sai là cả biển sai)
        return texts, float(min(confs))

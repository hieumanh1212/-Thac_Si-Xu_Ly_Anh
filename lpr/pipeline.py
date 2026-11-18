"""Ghép toàn bộ pipeline nhận dạng biển số xe máy và sinh ảnh minh hoạ từng bước."""

import time

import cv2
import numpy as np

from .char_cnn import CharClassifier
from .detect import find_plate_candidates
from .postprocess import is_valid, normalize_line1, normalize_line2, split_single_line
from .preprocess import preprocess
from .recognize import LINE1_ALLOW, LINE2_ALLOW, OCREngine, line_regions
from .segment import (PLATE_H, PLATE_W, binarize_plate, character_strip, compose_line,
                      mask_non_characters, score_candidate, segment_characters)
from .utils import resize_to_width, to_data_uri, warp_quad

WORK_WIDTH = 800  # chuẩn hoá chiều rộng ảnh trước khi xử lý


class LicensePlateRecognizer:
    def __init__(self, gpu=False):
        self.ocr = OCREngine(gpu=gpu)
        self.chars = CharClassifier()

    def warmup(self):
        self.chars.load()
        self.ocr.load()

    def status(self):
        return {"ocr_ready": self.ocr.ready, "ocr_loading": self.ocr.loading, "ocr_error": self.ocr.error,
                "cnn_available": self.chars.available}

    # ------------------------------------------------------------------
    def _evaluate(self, img, cand):
        plate = warp_quad(img, cand["quad"], (PLATE_W, PLATE_H))
        gray, binary, inverted = binarize_plate(plate)
        rows = segment_characters(binary)
        cand.update(plate=plate, gray=gray, binary=binary, inverted=inverted, rows=rows,
                    score=score_candidate(rows, cand["aspect"], cand["source"]))
        return cand

    def run(self, img_bgr, with_steps=True, ocr="auto"):
        """ocr: 'auto' (CNN ký tự nếu có mô hình, ngược lại EasyOCR) | 'cnn' | 'easyocr'."""
        t0 = time.perf_counter()
        img = resize_to_width(img_bgr, WORK_WIDTH)
        pre = preprocess(img)

        # --- Định vị + kiểm chứng ứng viên ---------------------------------
        cands = [self._evaluate(img, c) for c in find_plate_candidates(img, pre)]
        best = max(cands, key=lambda c: c["score"])
        n_chars = sum(len(r) for r in best["rows"])
        found = n_chars >= 4
        t_detect = time.perf_counter()

        # --- Nhận dạng ------------------------------------------------------
        raw1 = raw2 = ""
        conf1 = conf2 = 0.0
        regions = line_regions(best["rows"], best["gray"].shape)
        best["ocr_input"] = mask_non_characters(best["gray"], best["rows"])
        best["ocr_lines"] = []
        engine = "-"
        rows = best["rows"]
        # CNN chỉ dùng khi phân đoạn ra đúng cấu trúc biển xe máy: 2 dòng, 3-5 và 4-5 ký tự
        use_cnn = (ocr in ("auto", "cnn") and found and self.chars.available and len(rows) == 2
                   and 3 <= len(rows[0]) <= 5 and 4 <= len(rows[1]) <= 5)
        if use_cnn:
            texts, conf = self.chars.read_rows(best["gray"], rows)
            if texts is not None:
                raw1, raw2 = texts
                conf1 = conf2 = conf
                engine = "CNN"
                best["ocr_lines"] = [compose_line(best["gray"], r) for r in rows]
        if found and engine == "-":
            engine = "EasyOCR"
            if len(best["rows"]) == 2:
                # Ghép lại ký tự đã phân đoạn của từng dòng (cách đều, bỏ dấu '-' '.')
                lines = [compose_line(best["gray"], r) for r in best["rows"]]
            else:
                src = best["ocr_input"]
                lines = [src[y0:y1, x0:x1] for (y0, y1, x0, x1) in regions]
            best["ocr_lines"] = lines
            raw1, conf1 = self.ocr.read_line(lines[0], LINE1_ALLOW)
            raw2, conf2 = self.ocr.read_line(lines[1], LINE2_ALLOW)
            if raw1 and not raw2:
                raw1, raw2 = split_single_line(raw1)

        line1, line2 = normalize_line1(raw1), normalize_line2(raw2)
        t_end = time.perf_counter()

        result = {
            "found": found,
            "text": f"{line1} {line2}".strip(),
            "line1": line1,
            "line2": line2,
            "raw": [raw1, raw2],
            "confidence": round((conf1 + conf2) / 2, 3) if found else 0.0,
            "valid": is_valid(line1, line2),
            "num_chars": n_chars,
            "num_candidates": len(cands),
            "source": best["source"],
            "ocr_engine": engine,
            "time_ms": {
                "detect_segment": round((t_detect - t0) * 1000, 1),
                "ocr": round((t_end - t_detect) * 1000, 1),
                "total": round((t_end - t0) * 1000, 1),
            },
            "ocr_error": self.ocr.error,
        }

        if with_steps:
            result.update(self._visualize(img, pre, cands, best, regions, found))
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _visualize(img, pre, cands, best, regions, found):
        # Tất cả ứng viên (vàng) + ứng viên được chọn (xanh lá)
        cand_vis = img.copy()
        for c in cands:
            if c["source"] != "full":
                cv2.polylines(cand_vis, [c["quad"].astype(np.int32)], True, (0, 220, 255), 1)

        detected = img.copy()
        if found:
            cv2.polylines(detected, [best["quad"].astype(np.int32)], True, (0, 255, 0), 3)

        seg_vis = cv2.cvtColor(best["binary"], cv2.COLOR_GRAY2BGR)
        colors = [(0, 0, 255), (255, 128, 0)]
        for i, row in enumerate(best["rows"]):
            for (x, y, w, h) in row:
                cv2.rectangle(seg_vis, (x, y), (x + w, y + h), colors[i % 2], 2)

        line_vis = cv2.cvtColor(best["ocr_input"], cv2.COLOR_GRAY2BGR)
        for i, (y0, y1, x0, x1) in enumerate(regions):
            cv2.rectangle(line_vis, (x0, y0), (x1 - 1, y1 - 1), colors[i % 2], 2)

        strip = character_strip(best["binary"], best["rows"])

        # Các dòng thực sự đưa vào OCR, xếp chồng lên nhau
        ocr_lines = [ln for ln in best.get("ocr_lines", []) if ln is not None and ln.size]
        if ocr_lines:
            lw = max(ln.shape[1] for ln in ocr_lines)
            stacked = [cv2.copyMakeBorder(ln, 4, 4, 0, lw - ln.shape[1], cv2.BORDER_CONSTANT, value=255)
                       for ln in ocr_lines]
            ocr_vis = np.vstack(stacked)
        else:
            ocr_vis = line_vis

        steps = [
            ("Ảnh gốc", f"Chuẩn hoá chiều rộng {WORK_WIDTH}px", img),
            ("Ảnh xám", "Chuyển BGR → Grayscale", pre["gray"]),
            ("CLAHE", "Cân bằng histogram thích nghi, tăng tương phản cục bộ", pre["clahe"]),
            ("Lọc song phương", "Bilateral filter: khử nhiễu, giữ biên", pre["blur"]),
            ("Biên Canny", "Phát hiện biên (ngưỡng 30 / 200)", pre["edges"]),
            ("Phép đóng", "Morphology close nối biên đứt gãy", pre["closed"]),
            ("Vùng sáng (Otsu)", "Ngưỡng Otsu + phép mở tìm nền biển sáng", pre["bright"]),
            ("Ứng viên", f"{len(cands) - 1} contour có hình dạng giống biển số", cand_vis),
            ("Biển số được chọn", f"Điểm cao nhất (nguồn: {best['source']})", detected),
            ("Nắn phối cảnh", f"Perspective transform → {PLATE_W}×{PLATE_H}", best["plate"]),
            ("Nhị phân hoá", "Adaptive threshold (Gaussian)" + (" – đã đảo nền tối" if best["inverted"] else ""),
             best["binary"]),
            ("Phân đoạn ký tự", "Connected components + lọc kích thước, tách 2 dòng", seg_vis),
            ("Lọc ký tự", "Xoá dấu '-', '.', ốc vít, khung viền… chỉ giữ vùng ký tự", line_vis),
            ("Ảnh đưa vào OCR", "Ghép lại ký tự từng dòng, cách đều nhau", ocr_vis),
        ]
        out = {
            "steps": [{"title": t, "desc": d, "image": to_data_uri(im)} for t, d, im in steps],
            "detected_image": to_data_uri(detected),
            "plate_image": to_data_uri(best["plate"]),
            "chars_image": to_data_uri(strip, ".png") if strip is not None else None,
        }
        return out

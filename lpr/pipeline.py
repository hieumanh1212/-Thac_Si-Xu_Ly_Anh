"""Ghép toàn bộ pipeline nhận dạng biển số xe máy và sinh ảnh minh hoạ từng bước."""

import time

import cv2
import numpy as np

from .char_cnn import CharClassifier
from .detect import find_long_plate_candidates, find_plate_candidates
from .postprocess import (is_valid, is_valid_car, looks_like_car_series, normalize_car_series,
                          normalize_line1, normalize_line2, split_car_long, split_single_line)
from .preprocess import preprocess
from .recognize import LINE1_ALLOW, LINE2_ALLOW, OCREngine, line_regions
from .segment import (LONG_H, LONG_W, PLATE_H, PLATE_W, binarize_plate, character_strip, compose_line,
                      mask_non_characters, score_candidate, segment_characters, segment_single_row)
from .utils import resize_to_width, to_data_uri, warp_quad

WORK_WIDTH = 800  # chuẩn hoá chiều rộng ảnh trước khi xử lý

CAR_LONG_WARNING = "Đây là biển ô tô. Hệ thống được xây dựng cho biển xe máy nên kết quả chỉ mang tính tham khảo."
CAR_SQUARE_WARNING = ("Có thể đây là biển ô tô (dòng trên chỉ có mã tỉnh + chữ cái, không có số seri). "
                      "Hệ thống được xây dựng cho biển xe máy nên kết quả chỉ mang tính tham khảo.")


def _row_is_complete(top, bottom):
    """Dòng trên 3 ký tự có thật sự chỉ có 3 ký tự, hay do phân đoạn làm SÓT 1 ký tự của biển xe máy?

    - Sót ký tự ở GIỮA  -> có khoảng hở lớn giữa 2 ký tự liền kề.
    - Sót ký tự ở ĐẦU/CUỐI -> dòng trên bị LỆCH sang một bên so với dòng dưới.
      (Biển ô tô vuông "51G / 123.45": dòng trên ngắn hơn nhưng được CĂN GIỮA.)
    """
    widths = [b[2] for b in top]
    gaps = [top[i + 1][0] - (top[i][0] + top[i][2]) for i in range(len(top) - 1)]
    span_bot = bottom[-1][0] + bottom[-1][2] - bottom[0][0]
    center_top = (top[0][0] + top[-1][0] + top[-1][2]) / 2
    center_bot = (bottom[0][0] + bottom[-1][0] + bottom[-1][2]) / 2
    return (max(gaps) < 1.6 * float(np.median(widths))
            and abs(center_top - center_bot) < 0.06 * span_bot)


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

    def _evaluate_long(self, img, cand):
        """Kiểm chứng ứng viên biển ô tô dài: nắn về 520x110, tách 1 dòng, cần 7-10 ký tự cao đều."""
        plate = warp_quad(img, cand["quad"], (LONG_W, LONG_H))
        gray, binary, inverted = binarize_plate(plate)
        rows = segment_single_row(binary)
        score = -1.0
        # Biển số: chữ TỐI trên nền SÁNG. Loại dòng chữ ngày giờ của camera (chữ sáng trên nền tối)
        # và các vùng tối khác. inverted=True nghĩa là binarize_plate phát hiện nền tối.
        bright_bg = not inverted and float(np.mean(gray)) > 110
        if bright_bg and rows and 7 <= len(rows[0]) <= 10:
            hs = np.array([b[3] for b in rows[0]], dtype=float)
            ys = np.array([b[1] for b in rows[0]], dtype=float)
            spread = hs.std() / hs.mean() + ys.std() / hs.mean()
            if spread < 0.2:  # ký tự cao đều và thẳng hàng -> rất giống biển số
                score = len(rows[0]) - 10 * spread
        cand.update(plate=plate, gray=gray, binary=binary, inverted=inverted, rows=rows, score=score)
        return cand

    def run(self, img_bgr, with_steps=True, ocr="auto"):
        """ocr: 'auto' (CNN ký tự nếu có mô hình, ngược lại EasyOCR) | 'cnn' | 'easyocr'."""
        t0 = time.perf_counter()
        img = resize_to_width(img_bgr, WORK_WIDTH)
        pre = preprocess(img)

        # --- Định vị + kiểm chứng ứng viên ---------------------------------
        cands = [self._evaluate(img, c) for c in find_plate_candidates(img, pre)]
        best = max(cands, key=lambda c: c["score"])

        # Biển ô tô dài 1 dòng: nhánh xe máy không bắt được (tỉ lệ ~4.7) -> tìm riêng
        long_cands = [self._evaluate_long(img, c) for c in find_long_plate_candidates(img, pre)]
        long_ok = [c for c in long_cands if c["score"] > 0]
        is_long = bool(long_ok)
        if is_long:
            best = max(long_ok, key=lambda c: c["score"])
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

        if is_long:
            return self._finish_long(img, pre, cands + long_cands, best, regions, ocr, t0, t_detect, with_steps)

        # CNN chỉ dùng khi phân đoạn ra đúng cấu trúc biển 2 dòng: 3-5 và 4-5 ký tự
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

        # Biển 2 dòng mà dòng trên chỉ có "2 số + chữ cái" (vd. 51G) -> biển ô tô vuông
        is_car_square = (found and len(rows) == 2 and looks_like_car_series(raw1)
                         and _row_is_complete(rows[0], rows[1]))
        if is_car_square:
            line1, line2 = normalize_car_series(raw1), normalize_line2(raw2)
            text, valid = f"{line1}-{line2}", is_valid_car(line1, line2)
        else:
            line1, line2 = normalize_line1(raw1), normalize_line2(raw2)
            text, valid = f"{line1} {line2}".strip(), is_valid(line1, line2)
        t_end = time.perf_counter()

        result = self._result(
            found=found, text=text, line1=line1, line2=line2, raw=[raw1, raw2],
            confidence=(conf1 + conf2) / 2 if found else 0.0, valid=valid, n_chars=n_chars,
            n_cands=len(cands), best=best, engine=engine, times=(t0, t_detect, t_end),
            vehicle="car" if is_car_square else "motorbike", layout="two_rows",
            warning=CAR_SQUARE_WARNING if is_car_square else None)
        if with_steps:
            result.update(self._visualize(img, pre, cands, best, regions, found))
        return result

    def _finish_long(self, img, pre, cands, best, regions, ocr, t0, t_detect, with_steps):
        """Đọc biển ô tô dài 1 dòng (vd. 51G-123.45)."""
        row = best["rows"][0]
        raw, conf, engine = "", 0.0, "-"
        best["ocr_lines"] = [compose_line(best["gray"], row)]
        if ocr in ("auto", "cnn") and self.chars.available:
            texts, conf = self.chars.read_rows(best["gray"], [row])
            if texts is not None:
                raw, engine = texts[0], "CNN"
        if engine == "-":
            raw, conf = self.ocr.read_line(best["ocr_lines"][0], LINE1_ALLOW)
            engine = "EasyOCR"
        series, number = split_car_long(raw)
        series, number = normalize_car_series(series), normalize_line2(number)
        t_end = time.perf_counter()
        result = self._result(
            found=True, text=f"{series}-{number}", line1=series, line2=number, raw=[raw],
            confidence=conf, valid=is_valid_car(series, number), n_chars=len(row),
            n_cands=len(cands), best=best, engine=engine, times=(t0, t_detect, t_end),
            vehicle="car", layout="one_row", warning=CAR_LONG_WARNING)
        if with_steps:
            result.update(self._visualize(img, pre, cands, best, regions, True))
        return result

    def _result(self, *, found, text, line1, line2, raw, confidence, valid, n_chars, n_cands, best,
                engine, times, vehicle, layout, warning):
        t0, t_detect, t_end = times
        return {
            "found": found,
            "text": text,
            "line1": line1,
            "line2": line2,
            "raw": raw,
            "confidence": round(confidence, 3),
            "valid": valid,
            "num_chars": n_chars,
            "num_candidates": n_cands,
            "source": best["source"],
            "ocr_engine": engine,
            "vehicle": vehicle,      # "motorbike" | "car"
            "layout": layout,        # "two_rows" | "one_row"
            "warning": warning,      # cảnh báo hiển thị cho người dùng (None nếu là biển xe máy)
            "time_ms": {
                "detect_segment": round((t_detect - t0) * 1000, 1),
                "ocr": round((t_end - t_detect) * 1000, 1),
                "total": round((t_end - t0) * 1000, 1),
            },
            "ocr_error": self.ocr.error,
        }

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
            ("Biển số được chọn",
             "Biển ô tô dài 1 dòng (tỉ lệ ~4.7)" if best["source"] == "long"
             else f"Điểm cao nhất (nguồn: {best['source']})", detected),
            ("Nắn phối cảnh", "Perspective transform → "
             f"{best['plate'].shape[1]}×{best['plate'].shape[0]}", best["plate"]),
            ("Nhị phân hoá", "Adaptive threshold (Gaussian)" + (" – đã đảo nền tối" if best["inverted"] else ""),
             best["binary"]),
            ("Phân đoạn ký tự", "Connected components + lọc kích thước, "
             + ("1 dòng (biển ô tô dài)" if best["source"] == "long" else "tách 2 dòng"), seg_vis),
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

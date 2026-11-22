"""Ghép toàn bộ pipeline nhận dạng biển số xe máy và sinh ảnh minh hoạ từng bước."""

import time

import cv2
import numpy as np

from .char_cnn import CharClassifier
from .detect import find_long_plate_candidates, find_plate_candidates
from .postprocess import (get_province, is_valid, is_valid_car, looks_like_car_series, normalize_car_series,
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

    def _is_taillight(self, plate_bgr):
        """Kiểm tra và loại trừ đèn hậu / đèn phản quang màu đỏ."""
        hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, (0, 70, 50), (12, 255, 255))
        red2 = cv2.inRange(hsv, (168, 70, 50), (180, 255, 255))
        red_ratio = (np.count_nonzero(red1) + np.count_nonzero(red2)) / float(plate_bgr.shape[0] * plate_bgr.shape[1])
        return red_ratio > 0.10

    # ------------------------------------------------------------------
    def _evaluate(self, img, cand):
        plate = warp_quad(img, cand["quad"], (PLATE_W, PLATE_H))
        gray, binary, inverted = binarize_plate(plate)
        rows = segment_characters(binary)
        score = score_candidate(rows, cand["aspect"], cand["source"])
        if self._is_taillight(plate):
            score -= 10.0  # Khử triệt để đèn hậu màu đỏ
        cand.update(plate=plate, gray=gray, binary=binary, inverted=inverted, rows=rows, score=score)
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

    def _recognize_single_cand(self, c, ocr="auto"):
        """Nhận dạng ký tự và hậu xử lý cho 1 ứng viên biển số."""
        is_long = c.get("is_long", False)
        rows = c.get("rows", [])
        n_chars = sum(len(r) for r in rows)
        found = n_chars >= 4
        regions = line_regions(rows, c["gray"].shape)
        c["ocr_input"] = mask_non_characters(c["gray"], rows)
        c["ocr_lines"] = []
        engine = "-"
        raw1, raw2 = "", ""
        conf1, conf2 = 0.0, 0.0
        vehicle = "motorbike"
        layout = "two_rows"
        warning = None

        if is_long and rows:
            layout = "one_row"
            vehicle = "car"
            warning = CAR_LONG_WARNING
            row = rows[0]
            c["ocr_lines"] = [compose_line(c["gray"], row)]
            if ocr in ("auto", "cnn") and self.chars.available:
                texts, conf = self.chars.read_rows(c["gray"], [row])
                if texts is not None:
                    raw1, engine = texts[0], "CNN"
                    conf1 = conf2 = conf
            if engine == "-":
                raw1, conf1 = self.ocr.read_line(c["ocr_lines"][0], LINE1_ALLOW)
                conf2 = conf1
                engine = "EasyOCR"
            series, number = split_car_long(raw1)
            line1 = normalize_car_series(series)
            line2 = normalize_line2(number)
            text = f"{line1}-{line2}"
            valid = is_valid_car(line1, line2)
        else:
            use_cnn = (ocr in ("auto", "cnn") and found and self.chars.available and len(rows) == 2
                       and 3 <= len(rows[0]) <= 5 and 4 <= len(rows[1]) <= 6)
            if use_cnn:
                texts, conf = self.chars.read_rows(c["gray"], rows)
                if texts is not None:
                    raw1, raw2 = texts
                    conf1 = conf2 = conf
                    engine = "CNN"
                    c["ocr_lines"] = [compose_line(c["gray"], r) for r in rows]
            if found and engine == "-":
                engine = "EasyOCR"
                if len(rows) == 2:
                    lines = [compose_line(c["gray"], r) for r in rows]
                else:
                    src = c["ocr_input"]
                    lines = [src[y0:y1, x0:x1] for (y0, y1, x0, x1) in regions]
                c["ocr_lines"] = lines
                raw1, conf1 = self.ocr.read_line(lines[0], LINE1_ALLOW) if len(lines) > 0 else ("", 0.0)
                raw2, conf2 = self.ocr.read_line(lines[1], LINE2_ALLOW) if len(lines) > 1 else ("", 0.0)
                if raw1 and not raw2:
                    raw1, raw2 = split_single_line(raw1)

            is_car_square = (found and len(rows) == 2 and looks_like_car_series(raw1)
                             and _row_is_complete(rows[0], rows[1]))
            if is_car_square:
                vehicle = "car"
                warning = CAR_SQUARE_WARNING
                line1, line2 = normalize_car_series(raw1), normalize_line2(raw2)
                text, valid = f"{line1}-{line2}", is_valid_car(line1, line2)
            else:
                line1, line2 = normalize_line1(raw1), normalize_line2(raw2)
                text, valid = f"{line1} {line2}".strip(), is_valid(line1, line2)

        strip = character_strip(c["binary"], rows)
        return {
            "found": found,
            "text": text,
            "line1": line1,
            "line2": line2,
            "province": get_province(line1 or text),
            "raw": [raw1, raw2],
            "confidence": round((conf1 + conf2) / 2 if found else 0.0, 3),
            "valid": valid,
            "num_chars": n_chars,
            "source": c["source"],
            "ocr_engine": engine,
            "vehicle": vehicle,
            "layout": layout,
            "warning": warning,
            "bbox": [int(v) for v in c["bbox"]],
            "quad": c["quad"].astype(float).tolist(),
            "plate_image": to_data_uri(c["plate"]),
            "chars_image": to_data_uri(strip, ".png") if strip is not None else None,
            "cand": c,
            "regions": regions,
        }

    def run(self, img_bgr, with_steps=True, ocr="auto"):
        """Nhận dạng biển số (hỗ trợ cả ảnh đơn biển và ảnh đa biển số)."""
        t0 = time.perf_counter()
        img = resize_to_width(img_bgr, WORK_WIDTH)
        pre = preprocess(img)

        # --- Định vị + kiểm chứng ứng viên ---------------------------------
        cands = [self._evaluate(img, c) for c in find_plate_candidates(img, pre)]
        long_cands = [self._evaluate_long(img, c) for c in find_long_plate_candidates(img, pre)]
        for lc in long_cands:
            lc["is_long"] = True

        all_cands = cands + long_cands
        t_detect = time.perf_counter()

        # Lọc ứng viên đạt chuẩn (Multi-plate ALPR)
        all_cands.sort(key=lambda c: c["score"], reverse=True)
        from .utils import box_iou
        valid_cands = []
        for c in all_cands:
            n_c = sum(len(r) for r in c.get("rows", []))
            if c["score"] >= 1.0 and n_c >= 4 and c["source"] != "full":
                if all(box_iou(c["bbox"], v["bbox"]) < 0.35 for v in valid_cands):
                    valid_cands.append(c)

        if not valid_cands:
            best_cand = max(all_cands, key=lambda c: c["score"])
            valid_cands = [best_cand]

        # Nhận dạng từng biển số và lọc triệt để nhiễu rác (như chữ vụn E 21, bóng đèn)
        plates = []
        for c in valid_cands:
            p_res = self._recognize_single_cand(c, ocr=ocr)
            if not p_res["found"] or not p_res["text"]:
                continue
            clean_alnum = [ch for ch in p_res["text"] if ch.isalnum()]
            # Biển số xe VN luôn có tối thiểu 6-10 ký tự; loại bỏ các mảnh rác quá ngắn hoặc conf quá thấp
            if len(clean_alnum) < 6 or p_res["confidence"] < 0.18:
                continue
            plates.append(p_res)

        # Sắp xếp biển số từ trái sang phải theo x
        plates.sort(key=lambda p: p["bbox"][0])
        for idx, p in enumerate(plates):
            p["index"] = idx + 1

        t_end = time.perf_counter()

        # Biển đại diện chính (ưu tiên biển hợp lệ, conf cao nhất)
        best = max(plates, key=lambda p: (p["valid"], p["confidence"], p["cand"]["score"])) if plates else self._recognize_single_cand(valid_cands[0], ocr=ocr)
        found = bool(plates)

        out_plates = [
            {
                "index": p.get("index", i + 1),
                "text": p["text"],
                "line1": p["line1"],
                "line2": p["line2"],
                "province": p.get("province", get_province(p.get("line1", p.get("text")))),
                "raw": p["raw"],
                "confidence": p["confidence"],
                "valid": p["valid"],
                "num_chars": p["num_chars"],
                "vehicle": p["vehicle"],
                "layout": p["layout"],
                "ocr_engine": p["ocr_engine"],
                "warning": p["warning"],
                "bbox": p["bbox"],
                "plate_image": p["plate_image"],
                "chars_image": p["chars_image"],
            }
            for i, p in enumerate(plates)
        ]

        result = {
            "found": found,
            "plate_count": len(plates),
            "text": " · ".join(p["text"] for p in plates) if len(plates) > 1 else best["text"],
            "line1": best["line1"],
            "line2": best["line2"],
            "province": best.get("province", get_province(best.get("line1", best.get("text")))),
            "raw": best["raw"],
            "confidence": best["confidence"],
            "valid": best["valid"],
            "num_chars": best["num_chars"],
            "num_candidates": len(all_cands),
            "source": best["source"],
            "ocr_engine": best["ocr_engine"],
            "vehicle": best["vehicle"],
            "layout": best["layout"],
            "warning": best["warning"],
            "time_ms": {
                "detect_segment": round((t_detect - t0) * 1000, 1),
                "ocr": round((t_end - t_detect) * 1000, 1),
                "total": round((t_end - t0) * 1000, 1),
            },
            "plates": out_plates,
            "ocr_error": self.ocr.error,
        }

        if with_steps:
            vis = self._visualize_multi(img, pre, all_cands, plates, best)
            result.update(vis)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _visualize_multi(img, pre, cands, plates, best_plate):
        # Tất cả ứng viên (vàng)
        cand_vis = img.copy()
        for c in cands:
            if c["source"] != "full":
                cv2.polylines(cand_vis, [c["quad"].astype(np.int32)], True, (0, 220, 255), 1)

        # Ảnh phát hiện biển số (vẽ TẤT CẢ các biển tìm được màu xanh lá kèm nhãn)
        detected = img.copy()
        for idx, p in enumerate(plates):
            quad = np.array(p["quad"], dtype=np.int32)
            cv2.polylines(detected, [quad], True, (0, 255, 0), 3)

            bx, by, bw, bh = p["bbox"]
            label = f"#{p.get('index', idx+1)}: {p['text']}" if len(plates) > 1 else p["text"]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thick = 2
            (tw, th), baseline = cv2.getTextSize(label, font, scale, thick)
            ty = max(by - 6, th + 4)
            cv2.rectangle(detected, (bx, ty - th - 4), (bx + tw + 6, ty + baseline), (0, 200, 0), -1)
            cv2.putText(detected, label, (bx + 3, ty - 2), font, scale, (0, 0, 0), thick, cv2.LINE_AA)

        best = best_plate["cand"]
        regions = best_plate["regions"]
        colors = [(0, 0, 255), (255, 128, 0)]

        seg_vis = cv2.cvtColor(best["binary"], cv2.COLOR_GRAY2BGR)
        for i, row in enumerate(best["rows"]):
            for (x, y, w, h) in row:
                cv2.rectangle(seg_vis, (x, y), (x + w, y + h), colors[i % 2], 2)

        line_vis = cv2.cvtColor(best["ocr_input"], cv2.COLOR_GRAY2BGR)
        for i, (y0, y1, x0, x1) in enumerate(regions):
            cv2.rectangle(line_vis, (x0, y0), (x1 - 1, y1 - 1), colors[i % 2], 2)

        ocr_lines = [ln for ln in best.get("ocr_lines", []) if ln is not None and ln.size]
        if ocr_lines:
            lw = max(ln.shape[1] for ln in ocr_lines)
            stacked = [cv2.copyMakeBorder(ln, 4, 4, 0, lw - ln.shape[1], cv2.BORDER_CONSTANT, value=255)
                       for ln in ocr_lines]
            ocr_vis = np.vstack(stacked)
        else:
            ocr_vis = line_vis

        strip = character_strip(best["binary"], best["rows"])

        steps = [
            ("Ảnh gốc", f"Chuẩn hoá chiều rộng {WORK_WIDTH}px", img),
            ("Ảnh xám", "Chuyển BGR → Grayscale", pre["gray"]),
            ("CLAHE", "Cân bằng histogram thích nghi, tăng tương phản cục bộ", pre["clahe"]),
            ("Lọc song phương", "Bilateral filter: khử nhiễu, giữ biên", pre["blur"]),
            ("Biên Canny", "Phát hiện biên (ngưỡng 30 / 200)", pre["edges"]),
            ("Phép đóng", "Morphology close nối biên đứt gãy", pre["closed"]),
            ("Vùng sáng", "Ngưỡng thích nghi + Otsu tìm nền biển số sáng", pre.get("adapt", pre["bright"])),
            ("Ứng viên", f"{len(cands) - 1} contour có hình dạng giống biển số", cand_vis),
            ("Biển số phát hiện", f"Tìm thấy {len(plates)} biển số hợp lệ trên ảnh", detected),
            ("Nắn phối cảnh", f"Perspective transform biển chính → {best['plate'].shape[1]}×{best['plate'].shape[0]}", best["plate"]),
            ("Nhị phân hoá", "Adaptive threshold (Gaussian)" + (" – đã đảo nền tối" if best["inverted"] else ""), best["binary"]),
            ("Phân đoạn ký tự", f"Tách {len(best['rows'])} dòng ký tự thành phần liên thông", seg_vis),
            ("Lọc ký tự", "Xoá dấu '-', '.', ốc vít, khung viền… chỉ giữ vùng ký tự", line_vis),
            ("Ảnh đưa vào OCR", "Ghép lại ký tự từng dòng, cách đều nhau", ocr_vis),
        ]

        return {
            "steps": [{"title": t, "desc": d, "image": to_data_uri(im)} for t, d, im in steps],
            "detected_image": to_data_uri(detected),
            "plate_image": best_plate["plate_image"],
            "chars_image": best_plate["chars_image"],
        }

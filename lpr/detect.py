"""Bước 2 - Định vị vùng biển số.

Ý tưởng: tìm các contour trên ảnh biên và ảnh vùng sáng, giữ lại những contour có hình dạng
giống biển số xe máy (gần chữ nhật, tỉ lệ rộng/cao ~ 1.36 với biển 190x140 mm).
Mỗi ứng viên sau đó được kiểm chứng ở bước phân đoạn ký tự (pipeline chấm điểm).
"""

import cv2
import numpy as np

from .utils import box_iou, order_points, quad_size

# Tỉ lệ rộng / cao chấp nhận được của biển số xe máy (chuẩn ~1.36)
MIN_ASPECT, MAX_ASPECT = 0.9, 2.0
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.002, 0.85
MIN_FILL = 0.55  # diện tích contour / diện tích hình chữ nhật bao nhỏ nhất


def _candidates_from_mask(mask, source):
    h, w = mask.shape[:2]
    img_area = float(h * w)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_AREA_RATIO * img_area <= area <= MAX_AREA_RATIO * img_area):
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        fill = area / (rw * rh)
        if fill < MIN_FILL:
            continue

        # Xấp xỉ đa giác: nếu ra 4 đỉnh lồi thì dùng trực tiếp (giữ được phối cảnh),
        # ngược lại dùng hình chữ nhật xoay bao nhỏ nhất.
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype("float32")
        else:
            quad = cv2.boxPoints(rect).astype("float32")
        quad = order_points(quad)

        qw, qh = quad_size(quad)
        if qh < 10 or qw < 10:
            continue
        aspect = qw / qh
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue

        x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
        out.append({
            "quad": quad,
            "bbox": (x, y, bw, bh),
            "area": area,
            "aspect": aspect,
            "fill": fill,
            "source": source,
        })
    return out


def find_plate_candidates(img_bgr, pre, max_candidates=25):
    """Trả về danh sách ứng viên vùng biển số (đã loại trùng lặp)."""
    h, w = img_bgr.shape[:2]
    cands = _candidates_from_mask(pre["closed"], "edge") + _candidates_from_mask(pre["bright"], "bright")

    # Loại trùng: ưu tiên ứng viên có độ đầy (fill) cao hơn
    cands.sort(key=lambda c: c["fill"], reverse=True)
    unique = []
    for c in cands:
        if all(box_iou(c["bbox"], u["bbox"]) < 0.85 for u in unique):
            unique.append(c)
    unique.sort(key=lambda c: c["area"], reverse=True)
    unique = unique[:max_candidates]

    # Luôn thêm cả khung ảnh làm ứng viên: phục vụ trường hợp người dùng tải lên ảnh đã cắt sẵn biển số
    full = order_points(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32"))
    unique.append({
        "quad": full, "bbox": (0, 0, w, h), "area": float(w * h),
        "aspect": w / float(h), "fill": 1.0, "source": "full",
    })
    return unique

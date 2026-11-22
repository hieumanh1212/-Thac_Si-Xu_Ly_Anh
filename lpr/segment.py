"""Bước 3 - Nhị phân hoá biển số và phân đoạn ký tự.

    biển số (đã nắn) -> xám -> CLAHE -> (đảo nếu nền tối) -> ngưỡng thích nghi (adaptive threshold)
    -> phép mở khử nhiễu -> gán nhãn thành phần liên thông (connected components)
    -> lọc theo kích thước/tỉ lệ -> tách 2 dòng theo toạ độ y -> sắp xếp theo x
"""

import cv2
import numpy as np

# Kích thước chuẩn hoá của biển số xe máy sau khi nắn (tỉ lệ 190:140 mm)
PLATE_W, PLATE_H = 380, 280


def binarize_plate(plate_bgr):
    """Trả về (ảnh xám dùng cho OCR - chữ tối nền sáng, ảnh nhị phân - ký tự màu trắng)."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)

    # Biển nền tối chữ sáng (biển xanh, đỏ...) -> đảo ngược để luôn có chữ tối trên nền sáng
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = False
    if np.count_nonzero(otsu) < 0.45 * otsu.size:
        gray = 255 - gray
        inverted = True

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 10)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return gray, binary, inverted


# Kích thước chuẩn hoá biển ô tô dài 1 dòng (520 x 110 mm)
LONG_W, LONG_H = 520, 110


def segment_single_row(binary):
    """Phân đoạn ký tự trên biển ô tô dài 1 dòng. Trả về [row] hoặc [] (row sắp theo x)."""
    H, W = binary.shape[:2]
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (0.40 * H <= h <= 0.95 * H):   # ký tự chiếm phần lớn chiều cao biển dài
            continue
        if not (0.15 <= w / float(h) <= 1.0) or w > 0.14 * W:
            continue
        if not (0.15 <= area / float(w * h) <= 0.95):
            continue
        if y <= 1 or y + h >= H - 1:         # chạm mép trên/dưới -> viền
            continue
        boxes.append((x, y, w, h))
    if len(boxes) < 2:
        return []
    # Giữ các ký tự cao đều nhau (loại dấu '-', '.', ốc vít, logo)
    med_h = float(np.median([b[3] for b in boxes]))
    row = sorted((b for b in boxes if 0.75 * med_h <= b[3] <= 1.25 * med_h), key=lambda b: b[0])
    row = _remove_nested(row)
    return [row] if row else []


def segment_characters(binary):
    """Tìm hộp bao các ký tự. Trả về list các dòng, mỗi dòng là list (x, y, w, h) đã sắp theo x."""
    H, W = binary.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    def is_char(x, y, w, h, area):
        if not (0.12 * H <= h <= 0.52 * H):  # cho phép cả các mảnh ký tự bị đứt (vd nửa số 3)
            return False
        if not (0.012 * W <= w <= 0.35 * W):  # chữ M, W, K khá rộng
            return False
        if not (0.10 <= w / float(h) <= 1.40):
            return False
        if not (0.08 <= area / float(w * h) <= 0.98):
            return False
        # Chỉ loại nếu chạm sát mép viền ngoài cùng và bề rộng quá mỏng
        if (x <= 1 and w < 12) or (x + w >= W - 1 and w < 12):
            return False
        return True

    boxes = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if is_char(x, y, w, h, area):
            boxes.append((x, y, w, h))
        elif 0.16 * H <= h <= 0.52 * H and w > 0.8 * h:
            # Khối quá rộng: có thể là ký tự dính dấu '-' hoặc dính nhau -> tách bằng phép chiếu dọc
            comp = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
            for (px, py, pw, ph, pa) in _split_by_projection(comp):
                if is_char(x + px, y + py, pw, ph, pa):
                    boxes.append((x + px, y + py, pw, ph))

    if not boxes:
        return []

    # Tách dòng: tìm khoảng trống lớn nhất giữa các tâm y (có hỗ trợ biển nghiêng nhẹ)
    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    cys = [b[1] + b[3] / 2 for b in boxes]
    rows = [boxes]
    if len(boxes) > 1:
        gaps = np.diff(cys)
        k = int(np.argmax(gaps))
        if gaps[k] > 0.07 * H and 2 <= k + 1 <= len(boxes) - 2:
            rows = [boxes[: k + 1], boxes[k + 1:]]
        else:
            mid_y = H * 0.48
            r1 = [b for b in boxes if b[1] + b[3] / 2 < mid_y]
            r2 = [b for b in boxes if b[1] + b[3] / 2 >= mid_y]
            if len(r1) >= 2 and len(r2) >= 2:
                rows = [r1, r2]

    cleaned = []
    for ri, row in enumerate(rows):
        row.sort(key=lambda b: b[0])
        # Gộp các mảnh ký tự bị đứt (cả ngang như M, K lẫn dọc như số 3, 8 đứt eo)
        row = _merge_nearby_fragments(row, H, W)
        if not row:
            continue
        med_h = float(np.median([b[3] for b in row]))
        row = [b for b in row if 0.50 * med_h <= b[3] <= 1.50 * med_h]
        row = _remove_nested(row)
        row.sort(key=lambda b: b[0])
        # Giới hạn số lượng ký tự tối đa của dòng:
        # Dòng 1: tối đa 4 ký tự (loại bỏ viền / ốc vít thừa ở 2 bên)
        if ri == 0 and len(row) > 4:
            med_w = float(np.median([b[2] for b in row]))
            while len(row) > 4:
                worst_idx = max(range(len(row)), key=lambda idx: abs(row[idx][2] - med_w) + abs(row[idx][3] - med_h))
                row.pop(worst_idx)
                row.sort(key=lambda b: b[0])
        # Dòng 2: tối đa 5 ký tự (xe máy 4 hoặc 5 số)
        elif ri == 1 and len(row) > 5:
            med_w = float(np.median([b[2] for b in row]))
            while len(row) > 5:
                worst_idx = max(range(len(row)), key=lambda idx: abs(row[idx][2] - med_w) + abs(row[idx][3] - med_h))
                row.pop(worst_idx)
                row.sort(key=lambda b: b[0])
        if row:
            cleaned.append(row)
    return cleaned


def _merge_nearby_fragments(row, H, W):
    """Gộp các mảnh kề nhau hoặc chồng lấn thuộc về cùng 1 ký tự (vd: K, M đứt nét chéo, 3, 8 đứt eo)."""
    if len(row) < 2:
        return row
    char_w = max(float(np.median([b[2] for b in row])), 0.12 * W)
    char_h = max(float(np.median([b[3] for b in row])), 0.38 * H)
    changed = True
    while changed:
        changed = False
        new_row = []
        skip = set()
        for i in range(len(row)):
            if i in skip:
                continue
            ax, ay, aw, ah = row[i]
            merged = False
            for j in range(i + 1, len(row)):
                if j in skip:
                    continue
                bx, by, bw, bh = row[j]
                nx0, ny0 = min(ax, bx), min(ay, by)
                nx1, ny1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                mw, mh = nx1 - nx0, ny1 - ny0

                x_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
                min_w = min(aw, bw)

                # Trường hợp 1: Đứt dọc (số 3, 8 đứt eo giữa) -> x trùng nhau, y xếp chồng
                is_vertical_split = (min_w > 0 and x_overlap / min_w >= 0.35
                                     and mh <= 1.25 * char_h and mw <= 1.35 * char_w)

                # Trường hợp 2: Đứt ngang (chữ M, K, V đứt nét chéo) -> ít nhất 1 mảnh hẹp, x kề nhau
                x_gap = max(0, max(ax, bx) - min(ax + aw, bx + bw))
                is_horizontal_split = (x_gap <= 8 and (aw < 0.65 * char_w or bw < 0.65 * char_w)
                                       and mw <= 1.35 * char_w and mh <= 1.25 * char_h)

                if is_vertical_split or is_horizontal_split:
                    new_row.append((nx0, ny0, mw, mh))
                    skip.add(j)
                    skip.add(i)
                    changed = True
                    merged = True
                    break
            if not merged:
                new_row.append(row[i])
        row = sorted(new_row, key=lambda b: b[0])
    return row


def _split_by_projection(comp, band=0.2):
    """Tách dấu gạch ngang '-' bị dính vào ký tự bằng phép chiếu dọc.

    Dấu gạch ngang nằm ở giữa theo chiều cao: các cột của nó KHÔNG có điểm ảnh nào trong
    dải band*h phía trên và band*h phía dưới. Mọi cột của ký tự (kể cả nét chéo của K, M, N...)
    đều chạm ít nhất một trong hai dải đó, trừ một vài cột ngắn ở chỗ giao nhau.
    Trả về list (x, y, w, h, area) theo toạ độ của comp.
    """
    h = comp.shape[0]
    b = max(1, int(band * h))
    strong = comp[:b].any(axis=0) | comp[h - b:].any(axis=0)

    # Cột yếu ở hai đầu khối luôn bị cắt (dấu gạch dính một bên ký tự). Ở giữa khối chỉ cắt
    # khi đoạn yếu dài cỡ một dấu gạch hoàn chỉnh (dấu gạch nằm giữa hai ký tự).
    min_run = max(3, int(0.35 * h))
    W = len(strong)
    cx = 0
    while cx < W:
        if strong[cx]:
            cx += 1
            continue
        end = cx
        while end < W and not strong[end]:
            end += 1
        if cx > 0 and end < W and (end - cx) < min_run:
            strong[cx:end] = True
        cx = end
    pieces, start = [], None
    for cx, s in enumerate(list(strong) + [False]):
        if s and start is None:
            start = cx
        elif not s and start is not None:
            sub = comp[:, start:cx]
            ys = np.where(sub.any(axis=1))[0]
            if len(ys):
                y0, y1 = int(ys[0]), int(ys[-1]) + 1
                pieces.append((start, y0, cx - start, y1 - y0, int(sub.sum())))
            start = None
    return pieces


def _merge_fragments(row):
    """Gộp hai mảnh hẹp kề nhau thành một ký tự (chữ K, M, V... hay bị đứt ở nét chéo)."""
    if len(row) < 2:
        return row
    med_w = float(np.median([b[2] for b in row]))
    out = [row[0]]
    for b in row[1:]:
        a = out[-1]
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        h = max(ah, bh)
        gap = bx - (ax + aw)
        nx0, ny0 = min(ax, bx), min(ay, by)
        nx1, ny1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
        merged_w = nx1 - nx0
        narrow = aw < 0.6 * med_w or bw < 0.6 * med_w
        if narrow and gap <= 0.08 * h and merged_w <= 0.85 * h and merged_w <= 1.4 * med_w:
            out[-1] = (nx0, ny0, merged_w, ny1 - ny0)
        else:
            out.append(b)
    return out


def _remove_nested(boxes):
    """Bỏ các hộp nằm trọn trong hộp khác (vd. lỗ bên trong số 0, 8)."""
    keep = []
    for i, a in enumerate(boxes):
        ax, ay, aw, ah = a
        inside = False
        for j, b in enumerate(boxes):
            if i == j:
                continue
            bx, by, bw, bh = b
            if bx <= ax and by <= ay and ax + aw <= bx + bw and ay + ah <= by + bh and (bw * bh > aw * ah):
                inside = True
                break
        if not inside:
            keep.append(a)
    return keep


def mask_non_characters(gray, rows, pad=3):
    """Tô màu nền lên mọi vùng không phải ký tự (dấu '-', '.', ốc vít, khung viền...).

    OCR thường đọc nhầm dấu gạch ngang/dấu chấm thành chữ hoặc số, nên chỉ giữ lại
    các hộp ký tự đã phân đoạn.
    """
    if not rows:
        return gray
    H, W = gray.shape[:2]
    mask = np.zeros_like(gray)
    for row in rows:
        for (x, y, w, h) in row:
            cv2.rectangle(mask, (max(0, x - pad), max(0, y - pad)),
                          (min(W - 1, x + w + pad), min(H - 1, y + h + pad)), 255, -1)
    bg = int(np.percentile(gray[mask > 0], 90)) if np.any(mask) else 255
    out = np.full_like(gray, bg)
    out[mask > 0] = gray[mask > 0]
    return out


def compose_line(gray, row, pad=3):
    """Ghép lại các ký tự của một dòng, cách đều nhau, trên nền đồng nhất.

    Loại bỏ dấu '-', '.' và khoảng trống không đều giữa các nhóm ký tự, giúp OCR đọc ổn định hơn.
    """
    if not row:
        return None
    H, W = gray.shape[:2]
    h_max = max(b[3] for b in row) + 2 * pad
    gap = max(4, int(0.15 * h_max))
    crops = []
    for (x, y, w, h) in row:
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crops.append(gray[y0:y1, x0:x1])
    bg = int(np.percentile(np.concatenate([c.ravel() for c in crops]), 90))
    total_w = sum(c.shape[1] for c in crops) + gap * (len(crops) + 1)
    canvas = np.full((h_max + 2 * gap, total_w), bg, dtype=np.uint8)
    cx = gap
    for c in crops:
        ch, cw = c.shape[:2]
        oy = gap + (h_max - ch) // 2
        canvas[oy:oy + ch, cx:cx + cw] = c
        cx += cw + gap
    return canvas


def character_strip(binary, rows, cell=48):
    """Ghép các ký tự đã tách thành một dải ảnh để hiển thị."""
    tiles = []
    for row in rows:
        for (x, y, w, h) in row:
            ch = binary[y:y + h, x:x + w]
            side = max(w, h) + 8
            canvas = np.zeros((side, side), dtype=np.uint8)
            ox, oy = (side - w) // 2, (side - h) // 2
            canvas[oy:oy + h, ox:ox + w] = ch
            tile = cv2.resize(canvas, (cell, cell), interpolation=cv2.INTER_AREA)
            tile = cv2.copyMakeBorder(tile, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=128)
            tiles.append(tile)
    if not tiles:
        return None
    return np.hstack(tiles)


def score_candidate(rows, aspect, source):
    """Chấm điểm ứng viên biển số dựa trên cấu trúc ký tự tìm được.

    Biển xe máy VN: 2 dòng, dòng trên 3-5 ký tự (vd. 59X1), dòng dưới 4-5 chữ số (vd. 12345).
    """
    n = sum(len(r) for r in rows)
    s = 0.0
    if len(rows) == 2:
        s += 2.0
        top, bottom = len(rows[0]), len(rows[1])
        if 3 <= top <= 5:
            s += 1.5
        if 4 <= bottom <= 5:
            s += 1.5
    s += 0.4 * min(n, 9)
    if n > 10:
        s -= 0.6 * (n - 10)
    # Ký tự trong cùng một dòng phải cao đều và thẳng hàng
    for row in rows:
        if len(row) >= 2:
            hs = np.array([b[3] for b in row], dtype=float)
            ys = np.array([b[1] for b in row], dtype=float)
            s -= 4.0 * (hs.std() / hs.mean()) + 4.0 * (ys.std() / hs.mean())
    s -= 0.8 * abs(aspect - 1.36)
    if source == "full":
        s -= 1.0
    return s

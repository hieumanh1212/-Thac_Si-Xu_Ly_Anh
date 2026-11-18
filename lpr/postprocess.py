"""Bước 5 - Hậu xử lý: chuẩn hoá chuỗi OCR theo định dạng biển số xe máy Việt Nam.

Định dạng phổ biến:
    Dòng 1: [2 số mã tỉnh]-[1 chữ cái][1 số hoặc chữ]     vd. 59-X1, 29-B1, 29-AA (xe < 50cc)
    Dòng 2: 4 hoặc 5 chữ số                               vd. 1234, 123.45

OCR hay nhầm giữa chữ và số có hình dạng giống nhau (O/0, B/8, S/5...). Vì biết trước vị trí nào
phải là số, vị trí nào phải là chữ nên ta sửa lỗi theo vị trí.
"""

import re

DIGITS = "0123456789"
LETTERS = "ABCDEFGHKLMNPRSTUVXYZ"  # chữ cái dùng trên biển số VN (không có I, J, O, Q, W)

TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "C": "0", "I": "1", "J": "1", "L": "1",
            "T": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4"}
TO_LETTER = {"0": "D", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B", "7": "T", "1": "T"}

LINE1_RE = re.compile(r"^\d{2}-[A-Z][A-Z0-9]{0,2}$")
LINE2_RE = re.compile(r"^(\d{4}|\d{3}\.\d{2})$")


def _clean(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def normalize_line1(raw):
    s = list(_clean(raw)[:5])
    if len(s) < 3:
        return "".join(s)
    for i in (0, 1):
        s[i] = TO_DIGIT.get(s[i], s[i])
    s[2] = TO_LETTER.get(s[2], s[2])
    return "".join(s[:2]) + "-" + "".join(s[2:])


def normalize_line2(raw):
    s = "".join(TO_DIGIT.get(c, c) for c in _clean(raw))
    s = re.sub(r"\D", "", s)[:5]
    if len(s) == 5:
        return f"{s[:3]}.{s[3:]}"
    return s


def is_valid(line1, line2):
    return bool(LINE1_RE.match(line1 or "")) and bool(LINE2_RE.match(line2 or ""))


# ---------------------------------------------------------------------------
# Biển ô tô:  [2 số mã tỉnh][1-2 chữ cái seri]-[4 hoặc 5 chữ số]   vd. 51G-123.45, 30A-1234, 51LD-012.34
# - Biển dài 1 dòng: 51G-123.45
# - Biển vuông 2 dòng: dòng trên 51G, dòng dưới 123.45
# Khác biển xe máy: phần seri dòng trên KHÔNG có chữ số đi kèm (xe máy: 59-X1, 59-X12).
CAR_SERIES_RE = re.compile(r"^\d{2}[A-Z]{1,2}$")


def normalize_car_series(raw):
    """'51G' -> '51G' (sửa nhầm chữ/số theo vị trí, không chèn dấu '-')."""
    s = list(_clean(raw)[:4])
    if len(s) < 3:
        return "".join(s)
    for i in (0, 1):
        s[i] = TO_DIGIT.get(s[i], s[i])
    s[2] = TO_LETTER.get(s[2], s[2])
    return "".join(s)


def split_car_long(raw):
    """Tách chuỗi biển ô tô 1 dòng thành (seri, số).  '51G12345' -> ('51G', '12345')."""
    s = _clean(raw)
    m = re.match(r"^(\d{2}[A-Z]{1,2})(\d+)$", s)
    if m:
        return m.group(1), m.group(2)
    return (s[:3], s[3:]) if len(s) > 3 else (s, "")


def is_valid_car(series, number):
    return bool(CAR_SERIES_RE.match(series or "")) and bool(LINE2_RE.match(number or ""))


def looks_like_car_series(line1_raw):
    """Dòng trên của biển 2 dòng có dạng ô tô vuông (đúng 2 số + 1 chữ, vd. 51G) hay không.

    Không tính trường hợp 2 chữ cái (vd. 29-AB) vì biển xe máy 50cc cũng có dạng này.
    """
    return bool(re.match(r"^\d{2}[A-Z]$", _clean(line1_raw)))


def split_single_line(raw):
    """Khi OCR chỉ đọc được 1 chuỗi liền (vd. '59X112345'), tách thành 2 dòng."""
    s = _clean(raw)
    if len(s) >= 9:
        return s[:-5], s[-5:]
    if len(s) >= 7:
        return s[:-4], s[-4:]
    return s, ""

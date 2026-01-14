import math
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np

# CCPD label dictionaries
PROVINCES = [
    "皖",
    "沪",
    "津",
    "渝",
    "冀",
    "晋",
    "蒙",
    "辽",
    "吉",
    "黑",
    "苏",
    "浙",
    "京",
    "闽",
    "赣",
    "鲁",
    "豫",
    "鄂",
    "湘",
    "粤",
    "桂",
    "琼",
    "川",
    "贵",
    "云",
    "藏",
    "陕",
    "甘",
    "青",
    "宁",
    "新",
]

ALPHAS = list("ABCDEFGHJKLMNPQRSTUVWXYZ")  # I/O are not used
# CCPD 官方定义：字母在前，数字在后
ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ") + list("0123456789")
# CRNN 字符表：与训练时一致（字母在前）
CRNN_ALPHABET = PROVINCES + list("ABCDEFGHJKLMNPQRSTUVWXYZ") + list("0123456789")

# 后处理映射：修正因训练数据ADS顺序错误导致的字符错位
# 训练时用的错误ADS（数字在前） vs 正确ADS（字母在前）
_WRONG_ADS = list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
_CORRECT_ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
PLATE_FIX_MAP = {_WRONG_ADS[i]: _CORRECT_ADS[i] for i in range(len(_WRONG_ADS))}


def fix_plate_text(plate: str) -> str:
    """修正模型输出的车牌文本（第3-7位字符映射）"""
    if len(plate) <= 2:
        return plate
    # 前2位（省份+字母）不变，第3-7位做映射修正
    fixed = plate[:2]
    for c in plate[2:]:
        fixed += PLATE_FIX_MAP.get(c, c)
    return fixed


def decode_plate(label_part: str) -> str:
    """Decode CCPD label segment (e.g., '0_0_3_28_30_33_30_32') to plate text."""
    nums = [n for n in label_part.split("_") if n != ""]
    if len(nums) < 2:
        raise ValueError("label part too short")
    values = [int(n) for n in nums]
    plate = PROVINCES[values[0] % len(PROVINCES)]
    plate += ALPHAS[values[1] % len(ALPHAS)]
    for v in values[2:]:
        plate += ADS[v % len(ADS)]
    return plate


def _parse_point(chunk: str) -> Tuple[int, int]:
    x_str, y_str = chunk.split("&")
    return int(x_str), int(y_str)


def parse_ccpd_filename(path: Union[str, Path]) -> dict:
    """Parse CCPD filename to bbox, 4 points, plate text, and tilt angle."""
    name = Path(path).stem
    parts = name.split("-")
    if len(parts) < 5:
        raise ValueError(f"unexpected CCPD name: {name}")

    bbox_part = parts[2].split("_")
    if len(bbox_part) != 2:
        raise ValueError(f"unexpected bbox part: {parts[2]}")
    x1, y1 = _parse_point(bbox_part[0])
    x2, y2 = _parse_point(bbox_part[1])

    pts_part = parts[3].split("_")
    if len(pts_part) != 4:
        raise ValueError(f"expect 4 keypoints, got {len(pts_part)} in {name}")
    pts = [_parse_point(p) for p in pts_part]

    plate = decode_plate(parts[4])
    tilt = compute_tilt_angle(pts)
    return {"bbox": (x1, y1, x2, y2), "points": pts, "plate": plate, "tilt": tilt}


def compute_tilt_angle(points: Sequence[Tuple[float, float]]) -> float:
    """Compute absolute angle (deg) of the top edge formed by the first two points."""
    if len(points) < 2:
        return 0.0
    (x1, y1), (x2, y2) = points[0], points[1]
    rad = math.atan2(y2 - y1, x2 - x1)
    return abs(math.degrees(rad))


def order_points(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Order 4 points as tl, tr, br, bl."""
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_plate(
    image: np.ndarray,
    points: Sequence[Tuple[float, float]],
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Perspective warp to (out_w, out_h) using 4 source points."""
    src = order_points(points)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


def bbox_from_points(points: Sequence[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


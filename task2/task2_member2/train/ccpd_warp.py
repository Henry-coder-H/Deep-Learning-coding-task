import os
import cv2
import numpy as np

class CCPDWarp(object):
    def __init__(self, expand=1.08, min_size=10, rotate_if_tall=True, **kwargs):
        self.expand = float(expand)
        self.min_size = int(min_size)
        self.rotate_if_tall = bool(rotate_if_tall)

    def _parse_vertices(self, img_path):
        # ---- 1) img_path 类型兜底：不让它把 DataLoader 搞崩 ----
        try:
            if img_path is None:
                return None

            # 有些情况下可能传成 list/tuple（极少见，但先兜底）
            if isinstance(img_path, (list, tuple)) and len(img_path) > 0:
                img_path = img_path[0]

            if not isinstance(img_path, (str, bytes, os.PathLike)):
                return None

            img_path = os.fspath(img_path)
            if isinstance(img_path, bytes):
                img_path = img_path.decode("utf-8", errors="ignore")

            name = os.path.basename(img_path)
            stem, _ = os.path.splitext(name)
        except Exception:
            return 

        # ---- 2) 正常 CCPD 文件名解析 ----
        parts = stem.split("-")
        if len(parts) < 4:
            return None

        vert_str = parts[3]  # x&y_x&y_x&y_x&y (CCPD: starts from RB)
        pts = []
        try:
            for token in vert_str.split("_"):
                x, y = map(int, token.split("&"))
                pts.append([x, y])
        except Exception:
            return None

        pts = np.array(pts, dtype=np.float32)
        if pts.shape != (4, 2):
            return None

        # CCPD order: [RB, LB, LT, RT] -> [TL, TR, BR, BL]
        rb, lb, lt, rt = pts
        return np.array([lt, rt, rb, lb], dtype=np.float32)

    def __call__(self, data):
        img = data["image"]
        img_path = data.get("img_path", None)

        pts = self._parse_vertices(img_path)
        if pts is None:
            return data  # 关键：失败直接跳过，不要抛异常！

        c = pts.mean(axis=0)
        pts = (pts - c) * self.expand + c

        h, w = img.shape[:2]
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        tl, tr, br, bl = pts
        width = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
        height = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
        width = max(width, self.min_size)
        height = max(height, self.min_size)

        dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
        warped = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        if self.rotate_if_tall and warped.shape[0] / max(warped.shape[1], 1) >= 1.5:
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

        data["image"] = warped
        return data
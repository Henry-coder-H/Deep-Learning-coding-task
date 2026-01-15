import cv2
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR, TextRecognition

# --- 核心配置 (复刻自 test.py) ---
CONF_VEHICLE = 0.5         # 车辆检测阈值
MIN_CAR_W_FOR_OCR = 150    # 车宽小于此值不识别
CAR_OCR_UPSCALE = 3.0      # 车图放大倍数 (Det阶段)
PLATE_REC_UPSCALE = 2.5    # 车牌放大倍数 (Rec阶段)
ENABLE_WARP = True         # 启用透视变换

# 车辆类别 (COCO): car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 3, 5, 7}

# 正则过滤
CN_STRICT_RE = re.compile(r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}$")
FOREIGN_RE = re.compile(r"^(?=.*[A-Z])[A-Z0-9]{5,10}$")
VALID_CN_PREFIX = set(list("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼港澳使领警学挂"))

class PaddleVideoRecognizer:
    def __init__(self, yolo_path, use_gpu=True):
        """
        初始化：YOLO用于找车，PaddleOCR用于找牌和认字
        """
        device = "gpu:0" if use_gpu else "cpu"
        
        # 1. 加载车辆检测模型 (YOLOv8n)
        print(f"[PaddleModel] Loading YOLO from: {yolo_path}")
        self.det_vehicle = YOLO(yolo_path)

        # 2. 加载 PaddleOCR (Det + Rec)
        # 对应 test.py 中的配置
        self.ocr_pipe = PaddleOCR(
            ocr_version="PP-OCRv5",
            # show_log=False,
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True
        )
        self.rec_model = TextRecognition(model_name="PP-OCRv5_server_rec", device=device)
        print("[PaddleModel] Loaded successfully.")

    def recognize_image(self, img_bgr):
        """
        app.py 调用的统一接口
        输入: 当前帧 (BGR)
        输出: list [ {'bbox': [x1,y1,x2,y2], 'text': str, 'conf': float}, ... ]
        """
        results = []
        H, W = img_bgr.shape[:2]

        # 1. YOLO 找车
        yolo_res = self.det_vehicle.predict(img_bgr, conf=CONF_VEHICLE, verbose=False)[0]
        vehicle_boxes = self._filter_vehicle_boxes(yolo_res, W, H)

        # 2. 遍历每辆车
        for v_box in vehicle_boxes:
            vx1, vy1, vx2, vy2 = v_box
            
            # 尺寸过滤
            if (vx2 - vx1) < MIN_CAR_W_FOR_OCR:
                continue

            # 裁剪车辆区域
            car_crop = img_bgr[vy1:vy2, vx1:vx2]
            
            # 核心识别流程 (Det -> Warp -> Rec)
            text, score, _ = self._process_one_car(car_crop)

            if text:
                # 只要识别到了，就返回车辆的大框作为 bbox (方便 UI 绘制)
                # 注：如果需要精确的车牌小框，需要在此处进行坐标换算，
                # 但为了视频展示稳定性，通常返回车辆框视觉效果更好。
                results.append({
                    'bbox': [vx1, vy1, vx2, vy2], 
                    'text': text,
                    'conf': score
                })

        return results

    # --- 内部核心逻辑 (复刻自 test.py) ---

    def _process_one_car(self, car_crop):
        # 1. 放大车辆图 (为了更好地检测小车牌)
        car_in = car_crop
        if CAR_OCR_UPSCALE > 1.0:
            car_in = cv2.resize(car_in, None, fx=CAR_OCR_UPSCALE, fy=CAR_OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)

        # 2. Paddle Det: 找车牌四点框
        plate_crop, best_poly = self._find_best_plate(car_in)
        
        if plate_crop is None or plate_crop.size == 0:
            return None, 0.0, None

        # 3. 放大车牌图 (为了更好地识别文字)
        if PLATE_REC_UPSCALE > 1.0:
            plate_crop = cv2.resize(plate_crop, None, fx=PLATE_REC_UPSCALE, fy=PLATE_REC_UPSCALE, interpolation=cv2.INTER_CUBIC)

        # 4. Paddle Rec: 识别文字
        out0 = self.rec_model.predict(input=plate_crop)[0]
        j = out0.json.get("res", {}) if hasattr(out0, "json") else {}
        text = self._clean_text(j.get("rec_text", ""))
        score = float(j.get("rec_score", 0.0))
        
        if not text or not self._valid_candidate(text):
            return None, 0.0, None
            
        return text, score, best_poly

    def _find_best_plate(self, car_crop):
        """在车图里找最好的车牌框，并 Warp 出来"""
        res0 = self.ocr_pipe.predict(input=car_crop)[0]
        j = res0.json.get("res", {}) if hasattr(res0, "json") else {}
        texts = j.get("rec_texts", [])
        scores = j.get("rec_scores", [])
        polys = j.get("rec_polys", [])

        best_poly = None
        best_score = -1e9

        for t, sc, poly in zip(texts, scores, polys):
            if poly is None or len(poly) != 4: continue
            sc = float(sc)
            t2 = self._clean_text(t)
            if not t2: continue

            # 宽高比过滤
            pts = np.array(poly, dtype=np.int32)
            _, _, w, h = cv2.boundingRect(pts)
            ratio = w / (h + 1e-6)
            if ratio < 2.0 or ratio > 14.0: continue

            # 评分 (置信度 + 规则)
            q = self._quality(t2)
            s = sc + 0.05 * min(len(t2), 10) + 0.6 * q
            if q == 0: s -= 2.0
            if any("\u4e00" <= c <= "\u9fff" for c in t2) and (not self._cn_prefix_ok(t2)):
                s -= 2.0

            if s > best_score:
                best_score = s
                best_poly = poly

        if best_poly is None:
            return None, None

        if ENABLE_WARP:
            return self._warp_quad(car_crop, best_poly), best_poly
        
        # Fallback: simple crop
        pts = np.array(best_poly, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        return car_crop[y:y+h, x:x+w].copy(), best_poly

    # --- 辅助函数 ---
    def _filter_vehicle_boxes(self, r, W, H):
        if r.boxes is None or r.boxes.xyxy is None: return []
        xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(int)
        cls = r.boxes.cls.detach().cpu().numpy().astype(int) if r.boxes.cls is not None else None
        out = []
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            if int(cls[i]) in VEHICLE_CLASSES:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W-1, x2), min(H-1, y2)
                out.append((x1, y1, x2, y2))
        return out

    def _clean_text(self, s):
        return "".join(c for c in str(s).upper() if ("\u4e00"<=c<="\u9fff") or ("A"<=c<="Z") or ("0"<=c<="9"))

    def _cn_prefix_ok(self, t):
        return (len(t)>=1) and ("\u4e00"<=t[0]<="\u9fff") and (t[0] in VALID_CN_PREFIX)

    def _quality(self, t):
        if not t: return 0
        has_LD = lambda x: any("A"<=c<="Z" for c in x) and any("0"<=c<="9" for c in x)
        if CN_STRICT_RE.match(t) and self._cn_prefix_ok(t): return 4
        if self._cn_prefix_ok(t) and len(t)>=6 and has_LD(t): return 3
        if (not any("\u4e00"<=c<="\u9fff" for c in t)) and FOREIGN_RE.match(t) and has_LD(t): return 2
        return 1 if has_LD(t) else 0

    def _valid_candidate(self, t): return self._quality(t) >= 1

    def _warp_quad(self, img, quad):
        pts = np.array(quad, dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).reshape(-1)
        tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
        tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
        ordered = np.array([tl, tr, br, bl], dtype=np.float32)
        
        w = int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl)))
        h = int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr)))
        w, h = max(w, 10), max(h, 10)
        
        dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(img, M, (w, h))
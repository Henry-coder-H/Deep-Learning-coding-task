import cv2
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR, TextRecognition

# --- 核心配置 (保持原有) ---
CONF_VEHICLE = 0.5         # 车辆检测阈值
MIN_CAR_W_FOR_OCR = 150    # 车宽小于此值不识别
CAR_OCR_UPSCALE = 2.5      # 车图放大倍数 (Det阶段)
PLATE_REC_UPSCALE = 2    # 车牌放大倍数 (Rec阶段)
ENABLE_WARP = True         # 启用透视变换

# --- 新增：Test.py 中的记忆策略配置 ---
TRACKER = "bytetrack.yaml" # YOLO 追踪配置
FRAME_SKIP = 5             # 视频识别频率：每隔多少帧做一次OCR
IOU_ASSOC_THRESH = 0.4     # 记忆关联 IoU 阈值
MAX_MISS_FRAMES = 100      # 记忆保留最长帧数
UPGRADE_CONFIRM = 2        # 升级文本需要的连续命中次数
SCORE_DELTA_UPDATE = 0.03  # 同级替换需要的分数增量

# 车辆类别 (COCO): car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 3, 5, 7}

# 正则过滤
CN_STRICT_RE = re.compile(r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}$")
FOREIGN_RE = re.compile(r"^(?=.*[A-Z])[A-Z0-9]{5,10}$")
DIGITS_ONLY_RE = re.compile(r"^[0-9]{5,10}$") # 新增：纯数字正则用于过滤
VALID_CN_PREFIX = set(list("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼港澳使领警学挂"))

class PaddleVideoRecognizer:
    def __init__(self, yolo_path, use_gpu=True):
        """
        初始化：YOLO用于找车，PaddleOCR用于找牌和认字
        (严格保留原有代码)
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
        
        # [新增] 初始化记忆模块
        self.reset_memory()

    def reset_memory(self):
        """重置视频记忆状态 (切换视频时调用)"""
        self.memory = {}    # 存储 {mid: {info}}
        self.tid2mid = {}   # YOLO track_id -> memory_id
        self.next_mid = 1   # 下一个可用的 memory_id

    def recognize_image(self, img_bgr):
        """
        app.py 调用的统一接口 (图片模式)
        (保持原有逻辑不变，用于单张图片处理)
        """
        results = []
        H, W = img_bgr.shape[:2]

        # 1. YOLO 找车
        yolo_res = self.det_vehicle.predict(img_bgr, conf=CONF_VEHICLE, verbose=False)[0]
        vehicle_boxes = self._filter_vehicle_boxes(yolo_res, W, H)

        # 2. 遍历每辆车
        for v_box in vehicle_boxes:
            vx1, vy1, vx2, vy2 = v_box
            if (vx2 - vx1) < MIN_CAR_W_FOR_OCR: continue

            car_crop = img_bgr[vy1:vy2, vx1:vx2]
            text, score, _ = self._process_one_car(car_crop)

            if text:
                results.append({
                    'bbox': [vx1, vy1, vx2, vy2], 
                    'text': text,
                    'conf': score
                })
        return results

    def process_video_frame(self, img_bgr, frame_idx):
        """
        [新增] 视频专用处理接口：包含 追踪 + 记忆 + 择机OCR
        对应 test.py 中的 process_video 核心循环逻辑
        """
        H, W = img_bgr.shape[:2]
        
        # 1. YOLO Track (开启 persist=True 保持追踪)
        # 注意：这里使用 track 而不是 predict
        r = self.det_vehicle.track(img_bgr, persist=True, tracker=TRACKER, conf=CONF_VEHICLE, verbose=False)[0]
        dets = self._filter_vehicle_boxes(r, W, H, has_track_id=True) # 获取带 tid 的框

        # --- 以下是复刻 test.py 的记忆管理逻辑 ---

        # A. 所有记忆先记一次 "miss" (消失)
        for mid in list(self.memory.keys()):
            self.memory[mid]["miss"] += 1

        assigned = [None] * len(dets)
        used_mids = set()

        # B. 优先通过 YOLO track_id 找回记忆
        for i, d in enumerate(dets):
            tid = d["tid"]
            if tid is not None and tid in self.tid2mid:
                mid = self.tid2mid[tid]
                if mid in self.memory:
                    assigned[i] = mid
                    used_mids.add(mid)

        # C. 其次通过 IoU 找回 (防止 YOLO ID 切换或短暂丢失)
        for i, d in enumerate(dets):
            if assigned[i] is not None: continue
            box = d["xyxy"]
            best_mid, best_iou = None, 0.0
            for mid, m in self.memory.items():
                if mid in used_mids: continue
                v = self._iou(box, m["bbox"])
                if v > best_iou:
                    best_iou, best_mid = v, mid
            
            if best_mid is not None and best_iou >= IOU_ASSOC_THRESH:
                assigned[i] = best_mid
                used_mids.add(best_mid)

        # D. 为未匹配的车辆创建新记忆
        for i, d in enumerate(dets):
            if assigned[i] is None:
                mid = self.next_mid
                self.next_mid += 1
                self.memory[mid] = {
                    "bbox": d["xyxy"], 
                    "text": "", "score": 0.0, "miss": 0,
                    "pending_text": "", "pending_cnt": 0, "pending_score": 0.0
                }
                assigned[i] = mid

        # E. 更新被激活的记忆 (位置和 miss 计数)
        for i, d in enumerate(dets):
            mid = assigned[i]
            self.memory[mid]["bbox"] = d["xyxy"]
            self.memory[mid]["miss"] = 0
            if d["tid"] is not None:
                self.tid2mid[d["tid"]] = mid

        # F. 清理由于长时间消失而过期的记忆
        for mid in list(self.memory.keys()):
            if self.memory[mid]["miss"] > MAX_MISS_FRAMES:
                del self.memory[mid]
        # 同步清理映射表
        self.tid2mid = {tid: mid for tid, mid in self.tid2mid.items() if mid in self.memory}

        # --- 开始执行 OCR 策略 ---
        results = []
        do_ocr = (frame_idx % FRAME_SKIP == 0)

        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d["xyxy"]
            mid = assigned[i]
            
            # 策略：如果记忆里还没字，或者到了定时检查帧，并且车够宽 -> 做 OCR
            should_ocr = False
            has_text = bool(self.memory[mid]["text"])
            
            if (x2 - x1) >= MIN_CAR_W_FOR_OCR:
                if not has_text: should_ocr = True  # 没字，立即识别
                elif do_ocr: should_ocr = True      # 有字，例行检查

            if should_ocr:
                car_crop = img_bgr[y1:y2, x1:x2]
                if car_crop.size != 0:
                    text, score, _ = self._process_one_car(car_crop)
                    if text:
                        # 核心：使用打分逻辑更新记忆
                        self._update_memory_item(self.memory[mid], text, score)

            # --- 最终输出结果 ---
            # 直接从记忆中取最稳定的结果
            final_text = self.memory[mid].get("text", "")
            final_score = self.memory[mid].get("score", 0.0)

            if final_text:
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'text': final_text,
                    'conf': final_score,
                    'track_id': mid # 使用我们自己维护的 memory id
                })

        return results

    # --- 内部 OCR 核心逻辑 (复刻自 test.py) ---

    def _process_one_car(self, car_crop):
        # 1. 放大车辆图
        car_in = car_crop
        if CAR_OCR_UPSCALE > 1.0:
            car_in = cv2.resize(car_in, None, fx=CAR_OCR_UPSCALE, fy=CAR_OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)

        # 2. Paddle Det: 找车牌四点框
        plate_crop, best_poly = self._find_best_plate(car_in)
        
        if plate_crop is None or plate_crop.size == 0:
            return None, 0.0, None

        # 3. 放大车牌图
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

    # --- 辅助函数 (Logic from test.py) ---
    def _filter_vehicle_boxes(self, r, W, H, has_track_id=False):
        if r.boxes is None or r.boxes.xyxy is None: return []
        xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(int)
        cls = r.boxes.cls.detach().cpu().numpy().astype(int) if r.boxes.cls is not None else None
        # 如果是 track 模式，获取 ID
        ids = None
        if has_track_id and getattr(r.boxes, "id", None) is not None:
             ids = r.boxes.id.detach().cpu().numpy().astype(int)
             
        out = []
        for i, (x1, y1, x2, y2) in enumerate(xyxy):
            if int(cls[i]) in VEHICLE_CLASSES:
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W-1, x2), min(H-1, y2)
                
                if has_track_id:
                     tid = int(ids[i]) if ids is not None else None
                     out.append({"xyxy": (x1, y1, x2, y2), "tid": tid})
                else:
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

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0: return 0.0
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        return float(inter / (area_a + area_b - inter + 1e-6))
        
    def _should_update_same_level(self, old_t, old_s, new_t, new_s) -> bool:
        if new_s > old_s + SCORE_DELTA_UPDATE: return True
        if len(new_t) > len(old_t) and new_s >= old_s - 0.01: return True
        return False

    def _update_memory_item(self, m, new_text, new_score):
        """核心：更新记忆中的文字和分数 (Logic from test.py)"""
        if not new_text or not self._valid_candidate(new_text): return

        old_text = m.get("text", "")
        old_score = float(m.get("score", 0.0))
        old_q = self._quality(old_text)
        new_q = self._quality(new_text)

        # 1. 如果本来没字，直接赋值
        if not old_text:
            m["text"], m["score"] = new_text, float(new_score)
            m["pending_text"], m["pending_cnt"], m["pending_score"] = "", 0, 0.0
            return

        # 2. 如果新字质量差，放弃
        if new_q < old_q: return

        # 3. 如果质量同级，看分数和长度
        if new_q == old_q:
            if self._should_update_same_level(old_text, old_score, new_text, float(new_score)):
                m["text"], m["score"] = new_text, float(new_score)
            return

        # 4. 升级逻辑 (Level Up)：防止一帧误识别覆盖
        ptxt = m.get("pending_text", "")
        pcnt = int(m.get("pending_cnt", 0))
        pscore = float(m.get("pending_score", 0.0))

        if new_text == ptxt:
            pcnt += 1
            pscore = max(pscore, float(new_score))
        else:
            ptxt = new_text
            pcnt = 1
            pscore = float(new_score)

        m["pending_text"], m["pending_cnt"], m["pending_score"] = ptxt, pcnt, pscore

        if pcnt >= UPGRADE_CONFIRM:
            m["text"], m["score"] = ptxt, pscore
            m["pending_text"], m["pending_cnt"], m["pending_score"] = "", 0, 0.0

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
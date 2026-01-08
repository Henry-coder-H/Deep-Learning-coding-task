import os
import re
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR, TextRecognition
from PIL import Image, ImageDraw, ImageFont


INPUT_PATH = "input.png"   # 输入：图片(.jpg/.png/...) 或 视频(.mp4/.avi/...)
OUTPUT_NAME = "output"     
OUTPUT_DIR = "."          

ENABLE_WARP = True         # True: 用四点框做透视拉正；False: 仅水平bbox裁剪

FRAME_SKIP = 1             # 每隔多少帧尝试更新一次OCR缓存。1=每帧更新（最慢但易抓到清晰帧）
CONF_VEHICLE = 0.5         # YOLO车辆检测置信度阈值
MIN_CAR_W_FOR_OCR = 150    # 车框宽度小于该值时不更新OCR（越大越稳但远处车更难识别）

CAR_OCR_UPSCALE = 3.0      # 车框裁剪后，先放大再做OCR检测（提高中文细节；越大越慢）
PLATE_REC_UPSCALE = 2.5    # 车牌裁剪后，放大再做OCR识别（提高中文细节；越大越慢）

# “识别到就一直显示”的缓存策略
IOU_ASSOC_THRESH = 0.4     # IoU关联阈值：越大越严格，越小越容易续上但可能误关联
MAX_MISS_FRAMES = 100      # 车辆消失超过这么多帧，从记忆里移除

UPGRADE_CONFIRM = 2        # 升级到“更高等级文本”前，需要连续命中相同结果的次数
SCORE_DELTA_UPDATE = 0.03  # 同等级替换需要更高分的幅度（防抖）

YOLO_MODEL = "yolov8n.pt"
TRACKER = "bytetrack.yaml"
DEVICE_OCR = "gpu:0"

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_RATIO = 100           # 基础字号 ≈ min(W,H)/FONT_RATIO
FONT_MIN = 30
FONT_MAX = 260

THICKNESS_RATIO = 350      # 线宽 ≈ min(W,H)/THICKNESS_RATIO
THICKNESS_MIN = 2
THICKNESS_MAX = 14

DRAW_TEXT_BG = False       # True: 文字加黑底更清晰


# COCO车辆类别：car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASSES = {2, 3, 5, 7}

# 中国车牌：1中文 + 1字母 + 5/6位字母数字
CN_STRICT_RE = re.compile(r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}$")
# 国外/非中文：必须含字母（避免纯数字污染）
FOREIGN_RE = re.compile(r"^(?=.*[A-Z])[A-Z0-9]{5,10}$")
DIGITS_ONLY_RE = re.compile(r"^[0-9]{5,10}$")

# 合法省份/特殊前缀
VALID_CN_PREFIX = set(list("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼港澳使领警学挂"))


def is_image(p):
    return os.path.splitext(p)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]


def is_video(p):
    return os.path.splitext(p)[1].lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]


def out_path(inp, name, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    root, ext = os.path.splitext(name)
    if ext:
        return os.path.join(out_dir, name)
    if is_image(inp):
        in_ext = os.path.splitext(inp)[1].lower() or ".png"
        return os.path.join(out_dir, name + in_ext)
    return os.path.join(out_dir, name + ".mp4")


def clamp(x, a, b):
    return max(a, min(b, x))


def base_font(W, H):
    return int(clamp(min(W, H) / float(FONT_RATIO), FONT_MIN, FONT_MAX))


def thickness(W, H):
    return int(clamp(min(W, H) / float(THICKNESS_RATIO), THICKNESS_MIN, THICKNESS_MAX))


def font_for_box(fs_base, xyxy):
    x1, y1, x2, y2 = xyxy
    h = max(1, y2 - y1)
    fs_box = int(h * 0.10) # 10%盒高
    return int(clamp(max(fs_base, fs_box), FONT_MIN, FONT_MAX))


def clean_text(s: str) -> str:
    s = str(s).upper()
    return "".join(
        ch for ch in s
        if ("\u4e00" <= ch <= "\u9fff") or ("A" <= ch <= "Z") or ("0" <= ch <= "9")
    )


def has_letter_digit(t: str) -> bool:
    hasL = any("A" <= c <= "Z" for c in t)
    hasD = any("0" <= c <= "9" for c in t)
    return hasL and hasD


def cn_prefix_ok(t: str) -> bool:
    return (len(t) >= 1) and ("\u4e00" <= t[0] <= "\u9fff") and (t[0] in VALID_CN_PREFIX)


def quality(t: str) -> int:
    """
    返回等级（越大越好）：
      4: 严格中文车牌 且省份前缀合法
      3: 宽松中文候选（省份前缀合法 + 同时有字母数字 + 长度>=6）
      2: 字母数字混合（非中文）
      1: 只有字母（基本不太会用到）
      0: 其它（单汉字/纯数字等）
    """
    if not t:
        return 0
    if CN_STRICT_RE.match(t) and cn_prefix_ok(t):
        return 4
    if cn_prefix_ok(t) and len(t) >= 6 and has_letter_digit(t):
        return 3
    if (not any("\u4e00" <= c <= "\u9fff" for c in t)) and FOREIGN_RE.match(t) and has_letter_digit(t):
        return 2
    if any("A" <= c <= "Z" for c in t) and (not any("0" <= c <= "9" for c in t)) and (not any("\u4e00" <= c <= "\u9fff" for c in t)):
        return 1
    if DIGITS_ONLY_RE.match(t):
        return 0
    return 0


def valid_candidate(t: str) -> bool:
    # 关键：杜绝单汉字、以及纯数字进入缓存
    return quality(t) >= 1


def should_update_same_level(old_t, old_s, new_t, new_s) -> bool:
    # 同等级更新：更高分，或更长且分数不明显差
    if new_s > old_s + SCORE_DELTA_UPDATE:
        return True
    if len(new_t) > len(old_t) and new_s >= old_s - 0.01:
        return True
    return False


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_quad(img, quad, expand=1.10):
    q = np.array(quad, dtype=np.float32)
    c = q.mean(axis=0)
    q = (q - c) * expand + c
    q[:, 0] = np.clip(q[:, 0], 0, img.shape[1] - 1)
    q[:, 1] = np.clip(q[:, 1], 0, img.shape[0] - 1)

    q = order_points(q)
    tl, tr, br, bl = q
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    w, h = max(w, 10), max(h, 10)

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (w, h))


def best_plate_crop_in_car(car_crop, ocr_pipe):
    """
    在车框中用PPOCR det 找文字框，挑一个最像车牌的poly，然后裁/warp。
    """
    res0 = ocr_pipe.predict(input=car_crop)[0]
    j = res0.json.get("res", {}) if hasattr(res0, "json") else {}
    texts = j.get("rec_texts", [])
    scores = j.get("rec_scores", [])
    polys = j.get("rec_polys", [])

    best = None
    best_score = -1e9

    for t, sc, poly in zip(texts, scores, polys):
        if poly is None or len(poly) != 4:
            continue
        sc = float(sc)
        t2 = clean_text(t)
        if not t2:
            continue

        pts = np.array(poly, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        ratio = w / (h + 1e-6)
        if ratio < 2.0 or ratio > 14.0:
            continue

        q = quality(t2)
        # 选择策略：det置信度为主 + q加分；q=0（单汉字/纯数字）强降权
        s = sc + 0.05 * min(len(t2), 10) + 0.6 * q
        if q == 0:
            s -= 2.0
        # 中文但省份不合法也强降权（解决“广”）
        if any("\u4e00" <= c <= "\u9fff" for c in t2) and (not cn_prefix_ok(t2)):
            s -= 2.0

        if s > best_score:
            best_score = s
            best = poly

    if best is None:
        return None

    if ENABLE_WARP:
        return warp_quad(car_crop, best)

    # 不warp：用水平bbox裁剪（带pad）
    pts = np.array(best, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    pad = 6
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(car_crop.shape[1] - 1, x + w + pad)
    y2 = min(car_crop.shape[0] - 1, y + h + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return car_crop[y1:y2, x1:x2].copy()


def recognize_plate(car_crop, ocr_pipe, rec_model):
    """
    车框 ->（上采样）-> OCR det找牌 ->（上采样）-> OCR rec识别
    """
    if car_crop.shape[1] < MIN_CAR_W_FOR_OCR:
        return None, 0.0

    car_in = car_crop
    if CAR_OCR_UPSCALE > 1.0:
        car_in = cv2.resize(car_in, None, fx=CAR_OCR_UPSCALE, fy=CAR_OCR_UPSCALE, interpolation=cv2.INTER_CUBIC)

    plate = best_plate_crop_in_car(car_in, ocr_pipe)
    if plate is None or plate.size == 0:
        return None, 0.0

    if PLATE_REC_UPSCALE > 1.0:
        plate = cv2.resize(plate, None, fx=PLATE_REC_UPSCALE, fy=PLATE_REC_UPSCALE, interpolation=cv2.INTER_CUBIC)

    out0 = rec_model.predict(input=plate)[0]
    j = out0.json.get("res", {}) if hasattr(out0, "json") else {}
    text = clean_text(j.get("rec_text", ""))
    score = float(j.get("rec_score", 0.0))
    if not text:
        return None, 0.0
    return text, score


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-6))


def filter_vehicle_boxes(r, W, H):
    if r.boxes is None or r.boxes.xyxy is None:
        return []
    xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(int)
    cls = r.boxes.cls.detach().cpu().numpy().astype(int) if r.boxes.cls is not None else None
    ids = r.boxes.id.detach().cpu().numpy().astype(int) if getattr(r.boxes, "id", None) is not None else None

    out = []
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        c = int(cls[i]) if cls is not None else -1
        if c != -1 and c not in VEHICLE_CLASSES:
            continue
        tid = int(ids[i]) if ids is not None else None
        out.append({"xyxy": (int(x1), int(y1), int(x2), int(y2)), "tid": tid})
    return out


def draw_labels(img_bgr, labels):
    """
    labels: [(text, x, y, font_size), ...]
    代码短版：每帧转一次 PIL，循环画字。
    """
    if not labels:
        return img_bgr
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font_cache = {}

    for text, x, y, fs in labels:
        if fs not in font_cache:
            try:
                font_cache[fs] = ImageFont.truetype(FONT_PATH, fs)
            except Exception:
                font_cache[fs] = ImageFont.load_default()
        font = font_cache[fs]

        if DRAW_TEXT_BG:
            try:
                bx1, by1, bx2, by2 = draw.textbbox((x, y), text, font=font)
            except Exception:
                tw, th = draw.textsize(text, font=font)
                bx1, by1, bx2, by2 = x, y, x + tw, y + th
            pad = max(2, fs // 6)
            draw.rectangle([bx1 - pad, by1 - pad, bx2 + pad, by2 + pad], fill=(0, 0, 0))

        stroke = max(1, fs // 12)
        draw.text((x, y), text, font=font, fill=(255, 255, 0), stroke_width=stroke, stroke_fill=(0, 0, 0))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def update_memory_item(m, new_text, new_score):
    """
    m: {"text","score","pending_text","pending_cnt","pending_score"}
    规则：
      - 无效候选直接拒绝
      - 不降级
      - 同等级：按分数/长度更新
      - 升级：需要连续命中 UPGRADE_CONFIRM 次才提交（抑制一帧误升级）
    """
    if not new_text:
        return
    if not valid_candidate(new_text):
        return

    old_text = m.get("text", "")
    old_score = float(m.get("score", 0.0))
    old_q = quality(old_text)
    new_q = quality(new_text)

    if not old_text:
        m["text"], m["score"] = new_text, float(new_score)
        m["pending_text"], m["pending_cnt"], m["pending_score"] = "", 0, 0.0
        return

    # 不降级
    if new_q < old_q:
        return

    # 同等级：直接按分数/长度策略更新（用于修正字母数字）
    if new_q == old_q:
        if should_update_same_level(old_text, old_score, new_text, float(new_score)):
            m["text"], m["score"] = new_text, float(new_score)
        return

    # 升级：需要确认
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


def process_image(det, ocr_pipe, rec_model, img_bgr, output_path_):
    H, W = img_bgr.shape[:2]
    fs_base = base_font(W, H)
    th = thickness(W, H)

    annotated = img_bgr.copy()
    r = det.predict(img_bgr, conf=CONF_VEHICLE, verbose=False)[0]
    dets = filter_vehicle_boxes(r, W, H)

    labels = []
    for k, d in enumerate(dets):
        x1, y1, x2, y2 = d["xyxy"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), th)

        car_crop = img_bgr[y1:y2, x1:x2]
        text, score = recognize_plate(car_crop, ocr_pipe, rec_model)
        if text is None or (not valid_candidate(text)):
            label = f"ID:{k}"
        else:
            label = f"ID:{k} {text}({score:.2f})"

        fs = font_for_box(fs_base, (x1, y1, x2, y2))
        yy = y1 - (fs + 6)
        if yy < 0:
            yy = y1 + 2
        labels.append((label, x1 + 2, yy, fs))

    annotated = draw_labels(annotated, labels)
    ok = cv2.imwrite(output_path_, annotated)
    if not ok:
        raise RuntimeError(f"无法写出图片：{output_path_}")
    print("Saved:", output_path_)


def process_video(det, ocr_pipe, rec_model, video_path, output_path_):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fs_base = base_font(W, H)
    th = thickness(W, H)

    writer = cv2.VideoWriter(output_path_, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建视频写出器：{output_path_}")

    # memory: mid -> {"bbox","text","score","miss","pending_text","pending_cnt","pending_score"}
    memory = {}
    tid2mid = {}
    next_mid = 1

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        r = det.track(frame, persist=True, tracker=TRACKER, conf=CONF_VEHICLE, verbose=False)[0]
        dets = filter_vehicle_boxes(r, W, H)

        # miss++
        for mid in list(memory.keys()):
            memory[mid]["miss"] += 1

        # assign mid by tid hint then IoU
        assigned = [None] * len(dets)
        used = set()

        for i, d in enumerate(dets):
            tid = d["tid"]
            if tid is not None and tid in tid2mid and tid2mid[tid] in memory and tid2mid[tid] not in used:
                assigned[i] = tid2mid[tid]
                used.add(assigned[i])

        for i, d in enumerate(dets):
            if assigned[i] is not None:
                continue
            box = d["xyxy"]
            best_mid, best_i = None, 0.0
            for mid, m in memory.items():
                if mid in used:
                    continue
                v = iou(box, m["bbox"])
                if v > best_i:
                    best_i, best_mid = v, mid
            if best_mid is not None and best_i >= IOU_ASSOC_THRESH:
                assigned[i] = best_mid
                used.add(best_mid)

        for i, d in enumerate(dets):
            if assigned[i] is None:
                mid = next_mid
                next_mid += 1
                memory[mid] = {
                    "bbox": d["xyxy"], "text": "", "score": 0.0, "miss": 0,
                    "pending_text": "", "pending_cnt": 0, "pending_score": 0.0
                }
                assigned[i] = mid

        # update bbox/tid/miss
        for i, d in enumerate(dets):
            mid = assigned[i]
            memory[mid]["bbox"] = d["xyxy"]
            memory[mid]["miss"] = 0
            if d["tid"] is not None:
                tid2mid[d["tid"]] = mid

        # cleanup
        for mid in list(memory.keys()):
            if memory[mid]["miss"] > MAX_MISS_FRAMES:
                del memory[mid]
        tid2mid = {tid: mid for tid, mid in tid2mid.items() if mid in memory}

        annotated = frame.copy()
        labels = []
        do_ocr = (frame_idx % FRAME_SKIP == 0)

        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d["xyxy"]
            mid = assigned[i]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), th)

            if do_ocr and (x2 - x1) >= MIN_CAR_W_FOR_OCR:
                car_crop = frame[y1:y2, x1:x2]
                if car_crop.size != 0:
                    t, s = recognize_plate(car_crop, ocr_pipe, rec_model)
                    if t is not None:
                        update_memory_item(memory[mid], t, s)

            label = f"ID:{mid}"
            if memory[mid].get("text"):
                label += f" {memory[mid]['text']}({memory[mid]['score']:.2f})"

            fs = font_for_box(fs_base, (x1, y1, x2, y2))
            yy = y1 - (fs + 6)
            if yy < 0:
                yy = y1 + 2
            labels.append((label, x1 + 2, yy, fs))

        annotated = draw_labels(annotated, labels)
        writer.write(annotated)

        if frame_idx % 50 == 0 and frame_idx > 0:
            print(f"Processed frames: {frame_idx}")

    cap.release()
    writer.release()
    print("Saved:", output_path_)


def main():
    outp = out_path(INPUT_PATH, OUTPUT_NAME, OUTPUT_DIR)

    det = YOLO(YOLO_MODEL)
    ocr_pipe = PaddleOCR(
        ocr_version="PP-OCRv5",
        device=DEVICE_OCR,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    rec_model = TextRecognition(model_name="PP-OCRv5_server_rec", device=DEVICE_OCR)

    if is_image(INPUT_PATH):
        img = cv2.imread(INPUT_PATH)
        if img is None:
            raise RuntimeError(f"无法读取图片：{INPUT_PATH}")
        process_image(det, ocr_pipe, rec_model, img, outp)
        return

    if is_video(INPUT_PATH):
        process_video(det, ocr_pipe, rec_model, INPUT_PATH, outp)
        return

    # 不认识扩展名就尝试按视频打开
    cap = cv2.VideoCapture(INPUT_PATH)
    if cap.isOpened():
        cap.release()
        process_video(det, ocr_pipe, rec_model, INPUT_PATH, outp)
        return
    cap.release()

    raise RuntimeError(f"无法判断输入类型或文件不可读：{INPUT_PATH}")


if __name__ == "__main__":
    main()
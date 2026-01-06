import os, re, json, time, random, math
from pathlib import Path

import cv2
import numpy as np
import paddle
from paddleocr import TextRecognition  # 只用识别模型（Rec-only）
from PIL import Image, ImageDraw, ImageFont

SAVE_IMAGES = False   # True: 保存所有可视化；False: 仅保存最多50张错误可视化图

# 关闭可视化时，最多保存多少张错误图（全局上限，跨模式共用）
MAX_ERR_VIS = 50

# 模式： "nowarp" | "warp" | "both"
MODE = "both"  # 直接用 bbox 裁剪车牌 | 用四点透视矫正车牌 | 两种都跑一遍（可看提升对比）

# ========= 配置 =========
IMG_FOLDER = "./dataset/test"
OUT_DIR = "./visual_result_croprec"
MAX_N = 5000
DEVICE = "gpu:0"

RANDOM_SAMPLE = True
SEED = 0

# 裁剪时给 bbox 加一点 padding，避免框太紧截断字符
BBOX_PAD_RATIO = 0.08

# 鲁棒性透视畸变
ROBUST_PERSPECTIVE_QUANTILE = 0.90  # 分位数阈值

# CCPD 解码表（省份 + alphabets + ads）
PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
             "桂","琼","川","贵","云","藏","陕","甘","青","宁","新","警","学","O"]
ALPHABETS = list("ABCDEFGHJKL MNPQRSTUVWXYZ".replace(" ","")) + ["O"]
ADS = list("ABCDEFGHJKL MNPQRSTUVWXYZ".replace(" ","")) + list("0123456789") + ["O"]

STRICT_PLATE_RE = re.compile(r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}$")

# ========== 全局：错误可视化计数（SAVE_IMAGES=False时使用） ==========
ERR_VIS_COUNT = 0
ERR_VIS_DIR = None
# ===================================================================

def decode_ccpd_gt(img_name: str) -> str | None:
    stem = Path(img_name).stem
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    code = parts[4]
    try:
        idx = list(map(int, code.split("_")))
        if len(idx) < 7:
            return None
        prov = PROVINCES[idx[0]]
        alpha = ALPHABETS[idx[1]]
        rest = "".join(ADS[i] for i in idx[2:])  # 支持 5 或 6 位尾部
        return prov + alpha + rest
    except Exception:
        return None

def parse_ccpd_bbox_vertices(img_name: str):
    """
    CCPD 文件名：
    ...-bbox-vertices-plate-...
    bbox:    x1&y1_x2&y2
    vertices: x&y_x&y_x&y_x&y  (官方说明：四点从右下角开始给)
    """
    stem = Path(img_name).stem
    parts = stem.split("-")
    if len(parts) < 4:
        return None, None

    bbox_str = parts[2]
    vert_str = parts[3]

    x1y1, x2y2 = bbox_str.split("_")
    x1, y1 = map(int, x1y1.split("&"))
    x2, y2 = map(int, x2y2.split("&"))
    bbox = (x1, y1, x2, y2)

    pts = []
    for token in vert_str.split("_"):
        x, y = map(int, token.split("&"))
        pts.append([x, y])
    pts = np.array(pts, dtype=np.float32)  # (4,2)

    return bbox, pts

def clean_text(t: str) -> str:
    """只保留 汉字 / A-Z / 0-9，去掉 : · 空格 等"""
    t = str(t).upper()
    keep = []
    for ch in t:
        if ("\u4e00" <= ch <= "\u9fff") or ("A" <= ch <= "Z") or ("0" <= ch <= "9"):
            keep.append(ch)
    return "".join(keep)

def order_points(pts4: np.ndarray) -> np.ndarray:
    """将4点排序为 [tl, tr, br, bl]（用于 warp，保持你原逻辑不动）"""
    pts = pts4.copy()
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def order_points_for_geom(pts4: np.ndarray) -> np.ndarray:
    """
    将4点排序为 [tl, tr, br, bl]（用于几何打分更稳：diff 用 y-x）
    """
    pts = pts4.copy().astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)  # y - x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def perspective_score_from_vertices(pts4: np.ndarray, eps=1e-6) -> float | None:
    """
    用四点顶点计算“透视强度分数”（越大越强透视/越梯形）：
      score = |log(top/bottom)| + |log(left/right)|
    其中：
      top    = |tr - tl|
      bottom = |br - bl|
      left   = |bl - tl|
      right  = |br - tr|
    """
    if pts4 is None or pts4.shape != (4, 2):
        return None
    tl, tr, br, bl = order_points_for_geom(pts4)

    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))

    if min(top, bottom, left, right) < 1.0:
        return None

    s1 = abs(np.log((top + eps) / (bottom + eps)))
    s2 = abs(np.log((left + eps) / (right + eps)))
    return float(s1 + s2)

def crop_by_bbox(img_bgr, bbox, pad_ratio=BBOX_PAD_RATIO):
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)
    return img_bgr[y1:y2, x1:x2].copy()

def warp_by_vertices(img_bgr, pts4):
    """用四点做透视矫正，把车牌拉正"""
    rect = order_points(pts4)  # tl, tr, br, bl
    (tl, tr, br, bl) = rect

    def dist(a, b):
        return float(np.linalg.norm(a - b))

    widthA = dist(br, bl)
    widthB = dist(tr, tl)
    maxW = int(max(widthA, widthB))

    heightA = dist(tr, br)
    heightB = dist(tl, bl)
    maxH = int(max(heightA, heightB))

    maxW = max(maxW, 10)
    maxH = max(maxH, 10)

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH))
    return warped

def get_font(size=28):
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def draw_overlay_on_original(img_path, bbox, pts4, gt, pred, ok, save_path):
    """画 bbox + 四点 + GT/Pred"""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return

    # bbox（蓝色）
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 四点（黄色）
    if pts4 is not None:
        pts = np.array(pts4, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_bgr, [pts], True, (0, 255, 255), 2)

    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_font(28)

    draw.rectangle([0, 0, 900, 120], fill=(0, 0, 0))
    txt = f"GT:   {gt}\nPred: {pred}\n{'OK' if ok else 'ERR'}"
    draw.text((10, 10), txt, font=font, fill=(0, 255, 0) if ok else (255, 0, 0))

    out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, out)

def recog_plate(rec_model: TextRecognition, plate_bgr):
    """Rec-only 识别：返回 (pred, score, pred_raw)"""
    out = rec_model.predict(input=plate_bgr, batch_size=1)
    res = out[0].json.get("res", {})
    pred_raw = res.get("rec_text", "")
    score = float(res.get("rec_score", 0.0))
    pred = clean_text(pred_raw)
    return pred, score, pred_raw

def eval_one_mode(img_paths, rec_model, use_warp: bool, tag: str, persp_thr: float | None):
    """
    评测一遍：
    - nowarp: bbox 裁剪 + rec
    - warp:   四点透视 + rec
    鲁棒性：
    - 强透视子集：perspective_score >= persp_thr
    """
    global ERR_VIS_COUNT, ERR_VIS_DIR

    # 输出目录（SAVE_IMAGES=True 时）
    vis_dir = Path(OUT_DIR) / f"vis_{tag}"
    crop_dir = Path(OUT_DIR) / f"plate_{tag}"
    if SAVE_IMAGES:
        vis_dir.mkdir(parents=True, exist_ok=True)
        crop_dir.mkdir(parents=True, exist_ok=True)

    # 错误样本清单
    bad_cases_path = Path(OUT_DIR) / f"bad_cases_{tag}.jsonl"
    bad_cases_path.parent.mkdir(parents=True, exist_ok=True)
    bad_cases_path.write_text("", encoding="utf-8")

    total = full_ok = 0
    corr_chars = total_chars = 0
    time_sum = 0.0

    # 鲁棒性：强透视子集
    robust_total = 0
    robust_ok = 0

    # 显存峰值清零（可选）
    if DEVICE.startswith("gpu"):
        try:
            paddle.device.cuda.reset_max_memory_allocated()
        except Exception:
            pass

    for i, p in enumerate(img_paths, 1):
        gt = decode_ccpd_gt(p.name)
        if gt is None:
            continue

        bbox, pts4 = parse_ccpd_bbox_vertices(p.name)
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue

        t0 = time.perf_counter()

        if use_warp and pts4 is not None:
            plate = warp_by_vertices(img_bgr, pts4)
        else:
            plate = crop_by_bbox(img_bgr, bbox) if bbox is not None else img_bgr

        pred, score, pred_raw = recog_plate(rec_model, plate)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        time_sum += (t1 - t0)

        if not STRICT_PLATE_RE.match(pred):
            pred = ""

        ok = (pred == gt)
        total += 1
        full_ok += int(ok)

        n = len(gt)
        m = sum(1 for k in range(n) if (pred[k] if k < len(pred) else "") == gt[k])
        corr_chars += m
        total_chars += n

        # ===== 鲁棒性：强透视子集统计 =====
        if persp_thr is not None and pts4 is not None:
            ps = perspective_score_from_vertices(pts4)
            if ps is not None and ps >= persp_thr:
                robust_total += 1
                robust_ok += int(ok)

        # ===== 可视化保存规则 =====
        if SAVE_IMAGES:
            crop_name = f"{'OK' if ok else 'ERR'}_{i:05d}_{p.stem}.jpg"
            cv2.imwrite(str(crop_dir / crop_name), plate)

            vis_name = f"{'OK' if ok else 'ERR'}_{i:05d}_{p.name}"
            draw_overlay_on_original(
                str(p), bbox, pts4, gt,
                pred if pred else f"[INVALID]{pred_raw}",
                ok,
                str(vis_dir / vis_name)
            )
        else:
            # 只保存最多50张错误图（全局上限，跨模式共用）
            if (not ok) and (ERR_VIS_COUNT < MAX_ERR_VIS):
                ERR_VIS_COUNT += 1
                vis_name = f"ERR_{ERR_VIS_COUNT:03d}_{tag}_{p.name}"
                draw_overlay_on_original(
                    str(p), bbox, pts4, gt,
                    pred if pred else f"[INVALID]{pred_raw}",
                    False,
                    str(ERR_VIS_DIR / vis_name)
                )

        # 错误样本记录
        if not ok:
            ps = perspective_score_from_vertices(pts4) if pts4 is not None else None
            with bad_cases_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "img": str(p),
                    "gt": gt,
                    "pred": pred,
                    "pred_raw": pred_raw,
                    "score": float(score),
                    "use_warp": bool(use_warp),
                    "latency_ms": latency_ms,
                    "perspective_score": float(ps) if ps is not None else None
                }, ensure_ascii=False) + "\n")

        if i % 200 == 0 or i == 1:
            print(f"[{tag}] progress: {i}/{len(img_paths)} (SAVE_IMAGES={SAVE_IMAGES})")

    full_acc = full_ok / total if total else 0.0
    char_acc = corr_chars / total_chars if total_chars else 0.0
    avg_ms = (time_sum / total) * 1000.0 if total else 0.0
    fps = (total / time_sum) if time_sum > 0 else 0.0

    mem_peak_mb = 0.0
    if DEVICE.startswith("gpu"):
        try:
            mem_peak_mb = paddle.device.cuda.max_memory_allocated() / (1024 ** 2)
        except Exception:
            mem_peak_mb = -1.0

    robust_acc = (robust_ok / robust_total) if robust_total else 0.0

    return {
        "tag": tag,
        "total": total,
        "full_acc": full_acc,
        "char_acc": char_acc,
        "avg_ms": avg_ms,
        "fps": fps,
        "mem_peak_mb": mem_peak_mb,
        "robust_total": robust_total,
        "robust_acc": robust_acc,
        "vis_dir": str(vis_dir.resolve()) if SAVE_IMAGES else "",
        "crop_dir": str(crop_dir.resolve()) if SAVE_IMAGES else "",
        "bad_cases": str(bad_cases_path.resolve()),
    }

def main():
    global ERR_VIS_COUNT, ERR_VIS_DIR
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # SAVE_IMAGES=False 时：创建错误可视化目录
    if not SAVE_IMAGES:
        ERR_VIS_DIR = Path(OUT_DIR) / "vis_err_50"
        ERR_VIS_DIR.mkdir(parents=True, exist_ok=True)
        ERR_VIS_COUNT = 0
    else:
        ERR_VIS_DIR = None
        ERR_VIS_COUNT = 0

    if DEVICE.startswith("gpu"):
        try:
            paddle.device.set_device("gpu")
        except Exception:
            pass

    img_paths = sorted([p for p in Path(IMG_FOLDER).rglob("*.jpg")])
    if not img_paths:
        raise RuntimeError(f"No images under {IMG_FOLDER}")

    if RANDOM_SAMPLE:
        random.seed(SEED)
        random.shuffle(img_paths)

    img_paths = img_paths[:MAX_N] if MAX_N and MAX_N > 0 else img_paths

    print(f"Using {len(img_paths)} images. OUT_DIR={Path(OUT_DIR).resolve()} MODE={MODE} SAVE_IMAGES={SAVE_IMAGES}")

    # ===== 计算强透视阈值（一次，供 nowarp/warp 共用）=====
    persp_scores = []
    for p in img_paths:
        _, pts4 = parse_ccpd_bbox_vertices(p.name)
        ps = perspective_score_from_vertices(pts4)
        if ps is not None:
            persp_scores.append(ps)

    persp_thr = None
    if len(persp_scores) >= 10:
        persp_thr = float(np.quantile(np.array(persp_scores, dtype=np.float32), ROBUST_PERSPECTIVE_QUANTILE))

    print(f"[Robustness] strong perspective subset: quantile={ROBUST_PERSPECTIVE_QUANTILE:.2f}, "
          f"samples_with_score={len(persp_scores)}, threshold={persp_thr}")
    # ============================================

    rec = TextRecognition(model_name="PP-OCRv5_server_rec", device=DEVICE)

    # 预热
    bbox0, _ = parse_ccpd_bbox_vertices(img_paths[0].name)
    img0 = cv2.imread(str(img_paths[0]))
    plate0 = crop_by_bbox(img0, bbox0) if (img0 is not None and bbox0 is not None) else img0
    _ = rec.predict(input=plate0, batch_size=1)

    results = []
    if MODE in ("nowarp", "both"):
        results.append(eval_one_mode(img_paths, rec, use_warp=False, tag="nowarp", persp_thr=persp_thr))
    if MODE in ("warp", "both"):
        results.append(eval_one_mode(img_paths, rec, use_warp=True, tag="warp", persp_thr=persp_thr))

    # ====== 组装报告文本（终端输出 + 保存txt）======
    report_lines = []
    report_lines.append("==================== 最终评估报告（Rec-only）====================")
    report_lines.append(f"OUT_DIR: {Path(OUT_DIR).resolve()}")
    report_lines.append(f"SAVE_IMAGES: {SAVE_IMAGES}")
    if not SAVE_IMAGES:
        report_lines.append(f"错误可视化目录: {ERR_VIS_DIR.resolve()} (最多 {MAX_ERR_VIS} 张，实际保存 {ERR_VIS_COUNT} 张)")
    report_lines.append("")

    for r in results:
        report_lines.append(f"【模式：{r['tag']}】")
        if SAVE_IMAGES:
            report_lines.append(f"可视化目录(原图叠加): {r['vis_dir']}")
            report_lines.append(f"裁剪车牌图目录:       {r['crop_dir']}")
        report_lines.append(f"错误样本清单:         {r['bad_cases']}")
        report_lines.append(f"评测图片数量:         {r['total']}")
        report_lines.append("")
        report_lines.append("1) 准确率 (Accuracy)")
        report_lines.append(f"   - 全字匹配率（整牌全对）: {r['full_acc']*100:.2f}%")
        report_lines.append(f"   - 字符准确率（逐字符平均）: {r['char_acc']*100:.2f}%")
        report_lines.append("")
        report_lines.append("2) 推理速度 (Latency)")
        report_lines.append(f"   - 平均单张耗时: {r['avg_ms']:.2f} ms")
        report_lines.append(f"   - FPS:          {r['fps']:.2f}")
        report_lines.append("")
        report_lines.append("3) 显存占用 (Memory)")
        report_lines.append(f"   - 显存峰值: {r['mem_peak_mb']:.2f} MB")
        report_lines.append("")
        report_lines.append("4) 鲁棒性 (Robustness)")
        report_lines.append(f"   - 强透视子集整牌匹配率: {r['robust_acc']*100:.2f}% (样本数={r['robust_total']})")
        report_lines.append("")

    if MODE == "both" and len(results) == 2:
        nowarp = next(x for x in results if x["tag"] == "nowarp")
        warp = next(x for x in results if x["tag"] == "warp")
        gain = (warp["full_acc"] - nowarp["full_acc"]) * 100.0
        report_lines.append("【透视变换增益】")
        report_lines.append(f"   - 全字匹配率提升：{gain:+.2f}%  (warp - nowarp)")
        report_lines.append("")

    report_lines.append("=================================================================")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = Path(OUT_DIR) / "final_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[OK] 报告已保存到: {report_path.resolve()}")

if __name__ == "__main__":
    main()
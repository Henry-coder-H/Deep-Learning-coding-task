import os, re, json, time, random
from pathlib import Path

import cv2
import numpy as np
import paddle
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

SAVE_IMAGES = False     # True=可视化所有结果；False=仅可视化最多50张错误图
BATCH_PRINT = 100
MAX_ERR_VIS = 50        # SAVE_IMAGES=False 时，最多保存多少张错误可视化图

# 由模型自己找车牌在哪再做检测识别
# ========= 配置 =========
IMG_FOLDER = "./dataset/test"
OUT_DIR = "./visual_result"
MAX_N = 5000
DEVICE = "gpu:0"
RANDOM_SAMPLE = True
SEED = 0
# =======================

# CCPD 映射表（按 CCPD 规则：省份 + alphabets + ads）
PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
             "桂","琼","川","贵","云","藏","陕","甘","青","宁","新","警","学","O"]
ALPHABETS = list("ABCDEFGHJKL MNPQRSTUVWXYZ".replace(" ","")) + ["O"]
ADS = list("ABCDEFGHJKL MNPQRSTUVWXYZ".replace(" ","")) + list("0123456789") + ["O"]

# 7位普通：汉字+字母+5位；8位新能源：汉字+字母+6位
STRICT_PLATE_RE = re.compile(r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}$")

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
        rest = "".join(ADS[i] for i in idx[2:])  # 5或6位都支持
        return prov + alpha + rest
    except Exception:
        return None

def parse_ccpd_tilt(img_name: str):
    """返回 (h_tilt_deg, v_tilt_deg)，来自文件名第2段 tilt"""
    stem = Path(img_name).stem
    parts = stem.split("-")
    if len(parts) < 2:
        return None
    a, b = parts[1].split("_")
    return float(a), float(b)

def angle_to_abs_deg(x):
    """把 0~360 映射到 [-180,180] 后取绝对值。"""
    x = x % 360.0
    if x > 180.0:
        x -= 360.0
    return abs(x)

def clean_text(t: str) -> str:
    # 只保留 汉字 / A-Z / 0-9，去掉 : · 空格 等
    t = str(t).upper()
    keep = []
    for ch in t:
        if ("\u4e00" <= ch <= "\u9fff") or ("A" <= ch <= "Z") or ("0" <= ch <= "9"):
            keep.append(ch)
    return "".join(keep)

def pick_best_plate(rec_texts, rec_scores, rec_polys):
    """优先选满足车牌正则的候选；否则返回空（不要fallback成长句子）。"""
    best = ("", -1.0, None)
    for i in range(len(rec_texts)):
        t = clean_text(rec_texts[i])
        s = float(rec_scores[i]) if i < len(rec_scores) else 0.0
        poly = rec_polys[i] if i < len(rec_polys) else None
        if STRICT_PLATE_RE.match(t) and s > best[1]:
            best = (t, s, poly)
    return best  # 注意：不匹配就返回空字符串

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

def order_points(poly4):
    """仅用于可视化：让框看起来更规整（不影响识别）"""
    pts = np.array(poly4, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.array([tl, tr, br, bl], dtype=np.int32)

def draw_on_image(img_path, poly, gt, pred, ok, save_path):
    """保存可视化图片（框+文字），仅当调用时才执行"""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return

    color = (0, 255, 0) if ok else (0, 0, 255)

    if poly is not None:
        pts = np.array(poly, dtype=np.int32)
        if pts.shape == (4, 2):
            pts = order_points(pts).reshape((-1, 1, 2))
            cv2.polylines(img_bgr, [pts], True, color, 3)

    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_font(28)

    txt = f"GT:   {gt}\nPred: {pred}\n{'OK' if ok else 'ERR'}"
    draw.rectangle([0, 0, 900, 120], fill=(0, 0, 0))
    draw.text((10, 10), txt, font=font, fill=(0, 255, 0) if ok else (255, 0, 0))

    out = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, out)

def main():
    # 为了保存 report.txt +（可选）错误可视化，OUT_DIR 始终创建
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # 可视化输出目录：
    # - SAVE_IMAGES=True  => 保存所有图到 vis_all/
    # - SAVE_IMAGES=False => 只保存最多50张错误图到 vis_err_50/
    if SAVE_IMAGES:
        vis_dir = Path(OUT_DIR) / "vis_all"
    else:
        vis_dir = Path(OUT_DIR) / "vis_err_50"
    vis_dir.mkdir(parents=True, exist_ok=True)

    err_vis_count = 0  # SAVE_IMAGES=False 时用于限制最多50张错误图

    # 固定设备
    if DEVICE.startswith("gpu"):
        try:
            paddle.device.set_device("gpu")
        except Exception:
            pass

    # 关掉 doc 模块，避免加载 doc_ori / UVDoc
    ocr = PaddleOCR(
        ocr_version="PP-OCRv5",
        device=DEVICE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )

    # 收集图片
    img_paths = sorted([p for p in Path(IMG_FOLDER).rglob("*.jpg")])
    if not img_paths:
        raise RuntimeError(f"No images under {IMG_FOLDER}")

    if RANDOM_SAMPLE:
        random.seed(SEED)
        random.shuffle(img_paths)

    img_paths = img_paths[:MAX_N]
    print(f"Processing {len(img_paths)} images. SAVE_IMAGES={SAVE_IMAGES}, BATCH_PRINT={BATCH_PRINT}")

    # 预热
    _ = ocr.predict(str(img_paths[0]))

    # 统计指标
    total = 0
    full_ok = 0
    corr_chars = 0
    total_chars = 0
    time_sum = 0.0

    # 鲁棒性：倾斜>30度子集
    tilt_total = 0
    tilt_full_ok = 0

    # bad_cases 路径保持你的原逻辑：SAVE_IMAGES=False 时写到当前目录
    bad_cases_path = Path(OUT_DIR) / "bad_cases.jsonl" if SAVE_IMAGES else Path("bad_cases.jsonl")
    bad_cases_path.write_text("", encoding="utf-8")

    # 显存峰值清零
    if DEVICE.startswith("gpu"):
        try:
            paddle.device.cuda.reset_max_memory_allocated()
        except Exception:
            pass

    for i, p in enumerate(img_paths, 1):
        gt = decode_ccpd_gt(p.name)
        if gt is None:
            continue

        t0 = time.perf_counter()
        res = ocr.predict(str(p))[0]
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        time_sum += (t1 - t0)

        j = res.json.get("res", {})
        rec_texts = j.get("rec_texts", [])
        rec_scores = j.get("rec_scores", [])
        rec_polys  = j.get("rec_polys", [])

        pred, score, poly = pick_best_plate(rec_texts, rec_scores, rec_polys)

        ok = (pred == gt)
        total += 1
        full_ok += int(ok)

        # 字符准确率
        n = len(gt)
        m = sum(1 for k in range(n) if (pred[k] if k < len(pred) else "") == gt[k])
        corr_chars += m
        total_chars += n

        # 倾斜>30度鲁棒性
        tilt = parse_ccpd_tilt(p.name)
        if tilt is not None:
            h, v = tilt
            if max(angle_to_abs_deg(h), angle_to_abs_deg(v)) > 30:
                tilt_total += 1
                tilt_full_ok += int(ok)

        # 可视化保存规则：
        # - SAVE_IMAGES=True：保存所有结果
        # - SAVE_IMAGES=False：只保存最多50张错误图
        if SAVE_IMAGES:
            save_name = f"{'OK' if ok else 'ERR'}_{i:05d}_{p.name}"
            draw_on_image(str(p), poly, gt, pred, ok, str(vis_dir / save_name))
        else:
            if (not ok) and err_vis_count < MAX_ERR_VIS:
                err_vis_count += 1
                save_name = f"ERR_{err_vis_count:03d}_{p.name}"
                draw_on_image(str(p), poly, gt, pred, ok, str(vis_dir / save_name))

        # 错误样本记录
        if not ok:
            with bad_cases_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "img": str(p),
                    "gt": gt,
                    "pred": pred,
                    "score": float(score),
                    "rec_texts": [clean_text(x) for x in rec_texts],
                    "rec_scores": [float(x) for x in rec_scores],
                }, ensure_ascii=False) + "\n")

        # 批量打印进度
        if i % BATCH_PRINT == 0 or i == 1 or i == len(img_paths):
            cur_full = full_ok / total * 100 if total else 0.0
            cur_char = corr_chars / total_chars * 100 if total_chars else 0.0
            cur_ms = time_sum / total * 1000 if total else 0.0
            print(f"[进度 {i}/{len(img_paths)}] 当前全字={cur_full:.2f}% 字符={cur_char:.2f}% 平均耗时={cur_ms:.2f}ms")

    # 汇总指标
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

    tilt_acc = (tilt_full_ok / tilt_total) if tilt_total else 0.0

    # ===== 生成报告文本（同时打印+保存txt）=====
    report_lines = []
    report_lines.append("==================== 最终评估报告 ====================")
    report_lines.append(f"错误样本清单(bad_cases): {bad_cases_path.resolve()}")
    report_lines.append(f"评测图片数量: {total}")
    report_lines.append("")
    report_lines.append("1) 准确率 (Accuracy)")
    report_lines.append(f"   - 全字匹配率（整牌全对）: {full_acc*100:.2f}%")
    report_lines.append(f"   - 字符准确率（逐字符平均）: {char_acc*100:.2f}%")
    report_lines.append("")
    report_lines.append("2) 推理速度 (Latency)")
    report_lines.append(f"   - 平均单张耗时: {avg_ms:.2f} ms")
    report_lines.append(f"   - FPS（每秒处理张数）: {fps:.2f}")
    report_lines.append("")
    report_lines.append("3) 显存占用 (Memory)")
    report_lines.append(f"   - 显存峰值: {mem_peak_mb:.2f} MB")
    report_lines.append("")
    report_lines.append("4) 鲁棒性 (Robustness)")
    report_lines.append(f"   - 大角度倾斜(>30°)整牌匹配率: {tilt_acc*100:.2f}%   (样本数={tilt_total})")
    report_lines.append("======================================================")
    report_lines.append("")
    report_lines.append(f"可视化输出目录: {vis_dir.resolve()}")
    if not SAVE_IMAGES:
        report_lines.append(f"错误可视化保存张数: {err_vis_count} / {MAX_ERR_VIS}")

    report_text = "\n".join(report_lines)

    print("\n" + report_text)

    report_path = Path(OUT_DIR) / "final_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n[OK] 报告已保存到: {report_path.resolve()}")

if __name__ == "__main__":
    main()
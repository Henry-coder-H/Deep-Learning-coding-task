import os, re, json, time, random
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR, TextRecognition
from PIL import Image, ImageDraw, ImageFont

SAVE_IMAGES = False
BATCH_PRINT = 100
MAX_ERR_VIS = 50

IMG_FOLDER = "./dataset/test"
OUT_DIR = "./visual_result"
MAX_N = 5000
DEVICE = "gpu:0"
RANDOM_SAMPLE = True
SEED = 0

STRICT_PLATE_RE = re.compile(r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}$")
WARP_SCORE_DELTA = 0.08  # pred非空时：warp结果要比原score高出该值才替换；pred==""时不看此阈值

# 鲁棒性：强透视子集（用GT vertices算透视强度分数，取分位数阈值）
ROBUST_PERSPECTIVE_QUANTILE = 0.90  # 0.90=top10%最强透视样本

PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
             "桂","琼","川","贵","云","藏","陕","甘","青","宁","新","警","学","O"]
ALPHABETS = list("ABCDEFGHJKL MNPQRSTUVWXYZ".replace(" ", "")) + ["O"]
ADS = list("ABCDEFGHJKL MNPQRSTUVWXYZ".replace(" ", "")) + list("0123456789") + ["O"]

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def decode_ccpd_gt(img_name: str):
    stem = Path(img_name).stem
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    code = parts[4]
    try:
        idx = list(map(int, code.split("_")))
        if len(idx) < 7:
            return None
        return PROVINCES[idx[0]] + ALPHABETS[idx[1]] + "".join(ADS[i] for i in idx[2:])
    except Exception:
        return None


def clean_text(t: str) -> str:
    t = str(t).upper()
    return "".join(
        ch for ch in t
        if ("\u4e00" <= ch <= "\u9fff") or ("A" <= ch <= "Z") or ("0" <= ch <= "9")
    )


def pick_strict_plate(rec_texts, rec_scores, rec_polys):
    best_t, best_s, best_poly = "", -1.0, None
    for t, s, poly in zip(rec_texts, rec_scores, rec_polys):
        tt = clean_text(t)
        ss = float(s)
        if STRICT_PLATE_RE.match(tt) and ss > best_s:
            best_t, best_s, best_poly = tt, ss, poly
    return best_t, best_s, best_poly


def pick_poly_for_warp(rec_texts, rec_scores, rec_polys):
    best_poly, best = None, -1e9
    for t, s, poly in zip(rec_texts, rec_scores, rec_polys):
        if poly is None or len(poly) != 4:
            continue
        tt = clean_text(t)
        ss = float(s)

        x, y, w, h = cv2.boundingRect(np.array(poly, dtype=np.int32))
        ratio = w / (h + 1e-6)
        if ratio < 2.0 or ratio > 14.0:
            continue

        hasL = any("A" <= c <= "Z" for c in tt)
        hasD = any("0" <= c <= "9" for c in tt)

        sc = ss
        if 5 <= len(tt) <= 9: sc += 0.15
        if hasL: sc += 0.15
        if hasD: sc += 0.15
        if tt and ("\u4e00" <= tt[0] <= "\u9fff"): sc += 0.05

        if sc > best:
            best, best_poly = sc, poly
    return best_poly


def warp_plate_from_poly(img_bgr, poly4, expand=1.12):
    if poly4 is None:
        return None
    pts = np.array(poly4, dtype=np.float32)
    if pts.shape != (4, 2):
        return None

    c = pts.mean(axis=0)
    pts = (pts - c) * float(expand) + c
    pts[:, 0] = np.clip(pts[:, 0], 0, img_bgr.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, img_bgr.shape[0] - 1)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)  # y-x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    src = np.array([tl, tr, br, bl], dtype=np.float32)

    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    w, h = max(w, 10), max(h, 10)

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_bgr, M, (w, h))


def recog_rec_only(rec_model: TextRecognition, plate_bgr):
    out = rec_model.predict(input=plate_bgr, batch_size=1)
    res = out[0].json.get("res", {})
    pred = clean_text(res.get("rec_text", ""))
    score = float(res.get("rec_score", 0.0))
    return pred, score


def get_font(size=28):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_on_image(img_bgr, poly, gt, pred, ok, save_path, font):
    color = (0, 255, 0) if ok else (0, 0, 255)
    if poly is not None:
        pts = np.array(poly, dtype=np.int32)
        if pts.shape == (4, 2):
            cv2.polylines(img_bgr, [pts.reshape((-1, 1, 2))], True, color, 3)

    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([0, 0, 900, 120], fill=(0, 0, 0))
    txt = f"GT:   {gt}\nPred: {pred}\n{'OK' if ok else 'ERR'}"
    draw.text((10, 10), txt, font=font, fill=(0, 255, 0) if ok else (255, 0, 0))
    cv2.imwrite(save_path, cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))


def parse_vertices(img_name: str):
    stem = Path(img_name).stem
    parts = stem.split("-")
    if len(parts) < 4:
        return None
    vert_str = parts[3]
    try:
        pts = []
        for token in vert_str.split("_"):
            x, y = map(int, token.split("&"))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.float32)
        return pts if pts.shape == (4, 2) else None
    except Exception:
        return None


def perspective_score_from_vertices(pts4: np.ndarray, eps=1e-6):
    if pts4 is None or pts4.shape != (4, 2):
        return None
    pts = pts4.astype(np.float32)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)  # y-x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    top = float(np.linalg.norm(tr - tl))
    bottom = float(np.linalg.norm(br - bl))
    left = float(np.linalg.norm(bl - tl))
    right = float(np.linalg.norm(br - tr))
    if min(top, bottom, left, right) < 1.0:
        return None

    return float(abs(np.log((top + eps) / (bottom + eps))) + abs(np.log((left + eps) / (right + eps))))


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    vis_dir = Path(OUT_DIR) / ("vis_all" if SAVE_IMAGES else "vis_err_50")
    vis_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(Path(IMG_FOLDER).rglob("*.jpg"))
    if not img_paths:
        raise RuntimeError(f"No images under {IMG_FOLDER}")

    if RANDOM_SAMPLE:
        random.seed(SEED)
        random.shuffle(img_paths)

    img_paths = img_paths[:MAX_N]
    print(f"Processing {len(img_paths)} images. SAVE_IMAGES={SAVE_IMAGES}, BATCH_PRINT={BATCH_PRINT}")

    ocr = PaddleOCR(
        ocr_version="PP-OCRv5",
        device=DEVICE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    rec = TextRecognition(model_name="PP-OCRv5_server_rec", device=DEVICE)
    font = get_font(28)

    _ = ocr.predict(str(img_paths[0]))
    img0 = cv2.imread(str(img_paths[0]))
    if img0 is not None:
        _ = rec.predict(input=img0[:64, :256].copy(), batch_size=1)

    # 预计算透视分数 + 阈值
    persp_map = {}
    scores = []
    for p in img_paths:
        pts4 = parse_vertices(p.name)
        s = perspective_score_from_vertices(pts4)
        persp_map[str(p)] = s
        if s is not None:
            scores.append(s)
    persp_thr = float(np.quantile(np.array(scores, dtype=np.float32), ROBUST_PERSPECTIVE_QUANTILE)) if len(scores) >= 10 else None

    total = full_ok = 0
    corr_chars = total_chars = 0
    time_sum = 0.0

    robust_total = 0
    robust_ok = 0

    err_vis_count = 0
    bad_cases_path = Path(OUT_DIR) / "bad_cases.jsonl" if SAVE_IMAGES else Path("bad_cases.jsonl")
    bad_cases_path.write_text("", encoding="utf-8")

    for i, p in enumerate(img_paths, 1):
        gt = decode_ccpd_gt(p.name)
        if gt is None:
            continue

        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue

        t0 = time.perf_counter()

        res = ocr.predict(str(p))[0]
        j = res.json.get("res", {})
        rec_texts = j.get("rec_texts", [])
        rec_scores = j.get("rec_scores", [])
        rec_polys = j.get("rec_polys", [])

        pred, score, poly = pick_strict_plate(rec_texts, rec_scores, rec_polys)

        poly_w = poly if pred else pick_poly_for_warp(rec_texts, rec_scores, rec_polys)
        if poly_w is not None:
            warped = warp_plate_from_poly(img_bgr, poly_w, expand=1.12)
            if warped is not None and warped.size != 0:
                pred_w, score_w = recog_rec_only(rec, warped)
                if STRICT_PLATE_RE.match(pred_w):
                    if (not pred) or (float(score_w) > float(score) + WARP_SCORE_DELTA):
                        pred, score, poly = pred_w, float(score_w), poly_w

        t1 = time.perf_counter()
        time_sum += (t1 - t0)

        ok = (pred == gt)
        total += 1
        full_ok += int(ok)

        n = len(gt)
        corr_chars += sum(1 for k in range(n) if (pred[k] if k < len(pred) else "") == gt[k])
        total_chars += n

        ps = persp_map.get(str(p), None)
        if persp_thr is not None and ps is not None and ps >= persp_thr:
            robust_total += 1
            robust_ok += int(ok)

        if SAVE_IMAGES:
            save_name = f"{'OK' if ok else 'ERR'}_{i:05d}_{p.name}"
            draw_on_image(img_bgr.copy(), poly, gt, pred, ok, str(vis_dir / save_name), font)
        else:
            if (not ok) and err_vis_count < MAX_ERR_VIS:
                err_vis_count += 1
                save_name = f"ERR_{err_vis_count:03d}_{p.name}"
                draw_on_image(img_bgr.copy(), poly, gt, pred, ok, str(vis_dir / save_name), font)

        if not ok:
            with bad_cases_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "img": str(p),
                    "gt": gt,
                    "pred": pred,
                    "score": float(score),
                    "perspective_score": float(ps) if ps is not None else None,
                    "rec_texts": [clean_text(x) for x in rec_texts],
                    "rec_scores": [float(x) for x in rec_scores],
                }, ensure_ascii=False) + "\n")

        if i % BATCH_PRINT == 0 or i == 1 or i == len(img_paths):
            cur_full = full_ok / total * 100 if total else 0.0
            cur_char = corr_chars / total_chars * 100 if total_chars else 0.0
            cur_ms = time_sum / total * 1000 if total else 0.0
            print(f"[进度 {i}/{len(img_paths)}] 当前全字={cur_full:.2f}% 字符={cur_char:.2f}% 平均耗时={cur_ms:.2f}ms")

    full_acc = full_ok / total if total else 0.0
    char_acc = corr_chars / total_chars if total_chars else 0.0
    avg_ms = (time_sum / total) * 1000.0 if total else 0.0
    fps = (total / time_sum) if time_sum > 0 else 0.0
    robust_acc = (robust_ok / robust_total) if robust_total else 0.0

    report = "\n".join([
        "==================== 最终评估报告 ====================",
        f"错误样本清单(bad_cases): {bad_cases_path.resolve()}",
        f"评测图片数量: {total}",
        "",
        "1) 准确率 (Accuracy)",
        f"   - 全字匹配率（整牌全对）: {full_acc*100:.2f}%",
        f"   - 字符准确率（逐字符平均）: {char_acc*100:.2f}%",
        "",
        "2) 推理速度 (Latency)",
        f"   - 平均单张耗时: {avg_ms:.2f} ms",
        f"   - FPS（每秒处理张数）: {fps:.2f}",
        "",
        "4) 鲁棒性 (Robustness)",
        f"   - 强透视子集整牌匹配率: {robust_acc*100:.2f}%   (样本数={robust_total})",
        "======================================================",
        "",
        f"可视化输出目录: {vis_dir.resolve()}",
        (f"错误可视化保存张数: {err_vis_count} / {MAX_ERR_VIS}" if not SAVE_IMAGES else ""),
    ])

    print("\n" + report)
    report_path = Path(OUT_DIR) / "final_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[OK] 报告已保存到: {report_path.resolve()}")


if __name__ == "__main__":
    main()
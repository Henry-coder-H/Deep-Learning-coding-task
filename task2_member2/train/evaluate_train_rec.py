import os, re, json, time, shutil
from pathlib import Path

import cv2
import numpy as np
from paddleocr import TextRecognition
from PIL import Image, ImageDraw, ImageFont

DATASET_ROOT = Path("/home/zwm/datasets")
ALL_TEST = DATASET_ROOT / "splits/all_test.txt"
ALL_HARD = DATASET_ROOT / "splits/all_hardtest.txt"

# 识别推理模型目录（inference.json / inference.pdiparams / inference.yml）
REC_MODEL_DIR = Path("~/.paddlex/official_models/PP-OCRv5_server_rec_old").expanduser()

OUT_DIR = Path("./warp_rec_only_out")
DEVICE = "gpu:0"
MAX_N = None          # None = 全量；或填 5000 试跑
MAX_ERR_VIS = 50      # 只保存前50张错误可视化
ROTATE_IF_TALL = True # warp 后若过高（竖着），自动旋转90度
# =========================================================

STRICT_PLATE_RE = re.compile(r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}$")

PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
             "桂","琼","川","贵","云","藏","陕","甘","青","宁","新","警","学","O"]
ALPHABETS = list("ABCDEFGHJKL MNPQRSTUVWXYZ".replace(" ","")) + ["O"]
ADS = list("ABCDEFGHJKL MNPQRSTUVWXYZ".replace(" ","")) + list("0123456789") + ["O"]

def load_list(p: Path):
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out

def decode_ccpd_gt(img_name: str):
    stem = Path(img_name).stem
    parts = stem.split("-")
    if len(parts) < 5:
        return None
    code = parts[4]
    try:
        idx = list(map(int, code.split("_")))
        text = PROVINCES[idx[0]] + ALPHABETS[idx[1]] + "".join(ADS[i] for i in idx[2:])
        return text.rstrip("O")
    except Exception:
        return None

def clean_text(t: str) -> str:
    t = str(t).upper()
    return "".join(ch for ch in t if ("\u4e00" <= ch <= "\u9fff") or ("A" <= ch <= "Z") or ("0" <= ch <= "9"))

def parse_ccpd_bbox_vertices(img_name: str):
    stem = Path(img_name).stem
    parts = stem.split("-")
    if len(parts) < 4:
        return None, None

    # bbox: x1&y1_x2&y2
    bbox_str = parts[2]
    x1y1, x2y2 = bbox_str.split("_")
    x1, y1 = map(int, x1y1.split("&"))
    x2, y2 = map(int, x2y2.split("&"))
    bbox = (x1, y1, x2, y2)

    # vertices: x&y_x&y_x&y_x&y  (CCPD starts from RB)
    vert_str = parts[3]
    pts = []
    for token in vert_str.split("_"):
        x, y = map(int, token.split("&"))
        pts.append([x, y])
    pts = np.array(pts, dtype=np.float32)  # (4,2)

    return bbox, pts

def order_points_ccpd(pts4: np.ndarray) -> np.ndarray:
    # CCPD order: [RB, LB, LT, RT] -> [TL, TR, BR, BL]
    rb, lb, lt, rt = pts4
    return np.array([lt, rt, rb, lb], dtype=np.float32)

def warp_by_vertices(img_bgr, pts4, expand=1.08, min_size=10):
    if pts4 is None or pts4.shape != (4, 2):
        return None

    src = order_points_ccpd(pts4)

    # expand a bit
    c = src.mean(axis=0)
    src = (src - c) * float(expand) + c

    h, w = img_bgr.shape[:2]
    src[:, 0] = np.clip(src[:, 0], 0, w - 1)
    src[:, 1] = np.clip(src[:, 1], 0, h - 1)

    tl, tr, br, bl = src
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    out_w = max(out_w, min_size)
    out_h = max(out_h, min_size)

    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    warped = cv2.warpPerspective(img_bgr, M, (out_w, out_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if ROTATE_IF_TALL and warped.shape[0] / max(warped.shape[1], 1) >= 1.5:
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return warped

def get_font(size=28):
    cands = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for p in cands:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def draw_overlay_on_original(img_bgr, bbox, pts4, gt, pred, save_path, font):
    # bbox（蓝）
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # vertices（黄）
    if pts4 is not None and pts4.shape == (4,2):
        pts = np.array(pts4, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_bgr, [pts], True, (0, 255, 255), 2)

    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([0, 0, 1100, 120], fill=(0, 0, 0))
    draw.text((10, 10), f"GT:   {gt}\nPred: {pred}\nERR", font=font, fill=(255, 0, 0))
    cv2.imwrite(str(save_path), cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))

def recog_plate(rec_model: TextRecognition, plate_bgr):
    out = rec_model.predict(input=plate_bgr, batch_size=1)
    res = out[0].json.get("res", {})
    pred_raw = res.get("rec_text", "")
    score = float(res.get("rec_score", 0.0))
    pred = clean_text(pred_raw)
    return pred, score, pred_raw

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vis_dir = OUT_DIR / "vis_err_50"
    bad_cases = OUT_DIR / "bad_cases_warp.jsonl"
    report_path = OUT_DIR / "final_report.txt"

    # 每次运行清空可视化目录，避免“越跑越多”
    if vis_dir.exists():
        shutil.rmtree(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    bad_cases.write_text("", encoding="utf-8")

    test_list = load_list(ALL_TEST)
    hard_set = set(load_list(ALL_HARD))
    if MAX_N is not None:
        test_list = test_list[:MAX_N]

    print(f"Using split(all_test) images: {len(test_list)}")
    print(f"Hard subset(all_hardtest) size: {len(hard_set)}")
    print(f"REC_MODEL_DIR: {REC_MODEL_DIR.resolve()}")
    print(f"OUT_DIR: {OUT_DIR.resolve()}")

    # 关键：用 model_dir 指向导出模型目录
    rec = TextRecognition(model_dir=str(REC_MODEL_DIR), device=DEVICE)  # <!--citation:1-->
    font = get_font(28)

    total = full_ok = 0
    corr_chars = total_chars = 0
    hard_total = hard_ok = 0
    time_sum = 0.0
    err_vis_count = 0

    for i, rel in enumerate(test_list, 1):
        img_path = DATASET_ROOT / rel
        if not img_path.exists():
            continue

        gt = decode_ccpd_gt(img_path.name)
        if gt is None:
            continue

        bbox, pts4 = parse_ccpd_bbox_vertices(img_path.name)

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        t0 = time.perf_counter()
        plate = warp_by_vertices(img_bgr, pts4, expand=1.08, min_size=10)
        if plate is None or plate.size == 0:
            continue

        pred, score, pred_raw = recog_plate(rec, plate)
        t1 = time.perf_counter()
        time_sum += (t1 - t0)

        # 只接受“像车牌”的结果；否则视为失败
        if not STRICT_PLATE_RE.match(pred):
            pred_show = f"[INVALID]{clean_text(pred_raw)}"
            pred = ""
        else:
            pred_show = pred

        ok = (pred == gt)
        total += 1
        full_ok += int(ok)

        n = len(gt)
        corr_chars += sum(1 for k in range(n) if (pred[k] if k < len(pred) else "") == gt[k])
        total_chars += n

        if rel in hard_set:
            hard_total += 1
            hard_ok += int(ok)

        if not ok:
            with bad_cases.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "rel": rel,
                    "img": str(img_path),
                    "gt": gt,
                    "pred": pred,
                    "pred_raw": pred_raw,
                    "score": float(score),
                }, ensure_ascii=False) + "\n")

            if err_vis_count < MAX_ERR_VIS:
                err_vis_count += 1
                draw_overlay_on_original(
                    img_bgr.copy(), bbox, pts4, gt, pred_show,
                    vis_dir / f"ERR_{err_vis_count:03d}_{img_path.name}",
                    font
                )

        if i % 500 == 0:
            cur = full_ok / max(total, 1) * 100
            print(f"[{i}/{len(test_list)}] cur_full_acc={cur:.2f}%")

    full_acc = full_ok / max(total, 1)
    char_acc = corr_chars / max(total_chars, 1)
    hard_acc = hard_ok / max(hard_total, 1)
    avg_ms = (time_sum / max(total, 1)) * 1000.0
    fps = total / max(time_sum, 1e-6)

    report = "\n".join([
        "==================== 最终评估报告（Rec-only / GT-Warp）====================",
        f"错误样本清单(bad_cases): {bad_cases.resolve()}",
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
        f"   - all_hardtest 子集整牌匹配率: {hard_acc*100:.2f}%   (样本数={hard_total})",
        "======================================================",
        "",
        f"错误可视化输出目录: {vis_dir.resolve()}",
        f"错误可视化保存张数: {err_vis_count} / {MAX_ERR_VIS}",
    ])

    print("\n" + report)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[OK] 报告已保存到: {report_path.resolve()}")

if __name__ == "__main__":
    main()
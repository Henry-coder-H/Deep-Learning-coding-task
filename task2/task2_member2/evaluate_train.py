import os, re, json, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR

DATASET_ROOT = Path(r"D:/Study/2025-2026_fall/DL/code/task2/CCPD2019/CCPD2019")
ALL_TEST = DATASET_ROOT / "D:/Study/2025-2026_fall/DL/code/task2/CCPD2019/CCPD2019/splits/all_hardtest.txt"
ALL_HARD = DATASET_ROOT / "D:/Study/2025-2026_fall/DL/code/task2/CCPD2019/CCPD2019/splits/all_hardtest.txt"

DET_DIR = Path("C:/Users/24849/.paddlex/official_models/PP-OCRv5_server_det").expanduser()
REC_DIR = Path("D:/Study/2025-2026_fall/DL/code/Deep-Learning-coding-task/task2_member2/PP-OCRv5_server_rec").expanduser()

OUT_DIR = Path("./e2e_eval_out")
VIS_DIR = OUT_DIR / "vis_err"
BAD_JSONL = OUT_DIR / "bad_cases.jsonl"
OUT_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

MAX_ERR_VIS = 50
MAX_N = None
DEVICE = "gpu:0"   # cpu / gpu / gpu:0

STRICT_PLATE_RE = re.compile(r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{5,6}$")

# CCPD GT decode
PROVINCES = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤",
             "桂","琼","川","贵","云","藏","陕","甘","青","宁","新","警","学","O"]
ALPHABETS = list("ABCDEFGHJKL MNPQRSTUVWXYZ".replace(" ", "")) + ["O"]
ADS = list("ABCDEFGHJKL MNPQRSTUVWXYZ".replace(" ", "")) + list("0123456789") + ["O"]

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

def load_list(p: Path):
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    return out

def get_font(size=28):
    cands = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for fp in cands:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()

def draw_err(img_bgr, poly, gt, pred, save_path, font):
    if poly is not None:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_bgr, [pts], True, (0, 0, 255), 3)
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle([0, 0, 1200, 120], fill=(0, 0, 0))
    draw.text((10, 10), f"GT:   {gt}\nPred: {pred}\nERR", font=font, fill=(255, 0, 0))
    cv2.imwrite(str(save_path), cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR))

def pick_plate(rec_texts, rec_scores, rec_polys):
    best = ("", -1.0, None)
    fallback = ("", -1.0, None)
    for t, s, poly in zip(rec_texts, rec_scores, rec_polys):
        tt = clean_text(t)
        ss = float(s)
        if ss > fallback[1]:
            fallback = (tt, ss, poly)
        if STRICT_PLATE_RE.match(tt) and ss > best[1]:
            best = (tt, ss, poly)
    return best if best[0] else fallback

def main():
    test_list = load_list(ALL_TEST)
    hard_set = set(load_list(ALL_HARD))
    if MAX_N is not None:
        test_list = test_list[:MAX_N]

    pipeline = PaddleOCR(
        device=DEVICE,
        text_detection_model_dir=str(DET_DIR),
        text_recognition_model_dir=str(REC_DIR),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    BAD_JSONL.write_text("", encoding="utf-8")
    font = get_font(28)

    total = full_ok = 0
    corr_chars = total_chars = 0
    hard_total = hard_ok = 0
    err_vis_count = 0
    time_sum = 0.0

    for i, rel in enumerate(test_list, 1):
        img_path = DATASET_ROOT / rel
        if not img_path.exists():
            continue

        gt = decode_ccpd_gt(img_path.name)
        if gt is None:
            continue

        t0 = time.perf_counter()
        res0 = pipeline.predict(str(img_path))[0]
        j = res0.json.get("res", {})
        rec_texts = j.get("rec_texts", [])
        rec_scores = j.get("rec_scores", [])
        rec_polys  = j.get("rec_polys", [])
        t1 = time.perf_counter()
        time_sum += (t1 - t0)

        pred, score, poly = pick_plate(rec_texts, rec_scores, rec_polys)

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
            with BAD_JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"rel": rel, "img": str(img_path), "gt": gt, "pred": pred, "score": float(score)},
                                   ensure_ascii=False) + "\n")

            if err_vis_count < MAX_ERR_VIS:
                err_vis_count += 1
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None:
                    draw_err(img_bgr, poly, gt, pred, VIS_DIR / f"ERR_{err_vis_count:03d}_{img_path.name}", font)

        if i % 500 == 0:
            print(f"[{i}/{len(test_list)}] cur_full_acc={full_ok/max(total,1)*100:.2f}%")

    full_acc = full_ok / max(total, 1)
    char_acc = corr_chars / max(total_chars, 1)
    hard_acc = hard_ok / max(hard_total, 1)
    avg_ms = (time_sum / max(total, 1)) * 1000.0
    fps = total / max(time_sum, 1e-6)

    report = "\n".join([
        "==================== 最终评估报告 ====================",
        f"错误样本清单(bad_cases): {BAD_JSONL.resolve()}",
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
        f"错误可视化输出目录: {VIS_DIR.resolve()}",
        f"错误可视化保存张数: {err_vis_count} / {MAX_ERR_VIS}",
    ])

    print("\n" + report)
    (OUT_DIR / "final_report.txt").write_text(report, encoding="utf-8")
    print(f"\n[OK] 报告已保存到: {(OUT_DIR / 'final_report.txt').resolve()}")

if __name__ == "__main__":
    main()
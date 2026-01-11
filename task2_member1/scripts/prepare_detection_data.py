import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from pipeline.utils import parse_ccpd_filename  # noqa: E402


def link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            dst.unlink()
        except OSError:
            pass
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError as e:
            # Fallback
            if e.errno == 17:
                dst.unlink(missing_ok=True)
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)


def yolo_label_line(meta: dict, img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = meta["bbox"]
    pts = meta["points"]
    xc = ((x1 + x2) / 2) / img_w
    yc = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    kpts = []
    for px, py in pts:
        kpts.extend([px / img_w, py / img_h, 2])  # visible=2
    values = [0, xc, yc, bw, bh] + kpts
    return " ".join(f"{v:.6f}" if i > 0 else str(int(v)) for i, v in enumerate(values))


def prepare(
    train_txt: Path,
    val_txt: Path,
    ccpd_root: Path,
    out_root: Path,
    copy: bool = False,
) -> None:
    
    # Read file lists
    with open(train_txt, 'r') as f:
        train_lines = [line.strip() for line in f if line.strip()]
    with open(val_txt, 'r') as f:
        val_lines = [line.strip() for line in f if line.strip()]

    # Process each subset
    for subset, lines in [("train", train_lines), ("val", val_lines)]:
        img_dir = out_root / "images" / subset
        label_dir = out_root / "labels" / subset
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for line in tqdm(lines, desc=f"{subset}"):
            # The line is relative path like "ccpd_rotate/0742..."
            # We need to find this file under ccpd_root or its subfolders if structure differs
            # Assuming line matches structure under raw_ccpd/CCPD2019 or raw_ccpd/CCPD2020
            # Try to resolve absolute path
            
            # Strategy: check if file exists directly under ccpd_root/line
            # Or search recursively if needed. 
            # But "all_train.txt" usually implies relative path from dataset root.
            # User said: "CCPD2019" and "CCPD2020" are under "raw_ccpd"
            # But the txt file content shows "ccpd_rotate/..." which belongs to CCPD2019/2020
            # We will try to find the file in either CCPD2019 or CCPD2020
            
            found_path = None
            # Check widely common paths
            potential_roots = [
                ccpd_root, 
                ccpd_root / "CCPD2019", 
                ccpd_root / "CCPD2020",
                ccpd_root / "CCPD2019" / "ccpd_rotate" # in case the txt just has filename
            ]
            
            # Simple check: join ccpd_root with line
            # The txt lines are like "ccpd_rotate/xxx.jpg"
            # So we check raw_ccpd/CCPD2019/ccpd_rotate/xxx.jpg etc.
            
            for root_candidate in [ccpd_root / "CCPD2019", ccpd_root / "CCPD2020"]:
                p = root_candidate / line
                if p.exists():
                    found_path = p
                    break
            
            if not found_path:
                # Fallback: maybe the txt file has absolute paths or different structure?
                # Try just ccpd_root/line
                p = ccpd_root / line
                if p.exists():
                    found_path = p
            
            if not found_path:
                # print(f"[WARN] Cannot find {line}")
                continue

            try:
                meta = parse_ccpd_filename(found_path.name)
            except Exception as exc:
                continue

            img = cv2.imread(str(found_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            label = yolo_label_line(meta, w, h)

            dst_img = img_dir / found_path.name
            dst_lbl = label_dir / f"{found_path.stem}.txt"
            link_or_copy(found_path, dst_img, copy=copy)
            dst_lbl.write_text(label + "\n", encoding="utf-8")

    print(f"[INFO] labels saved to {out_root/'labels'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CCPD to YOLOv8 pose format using specified train/test lists."
    )
    parser.add_argument("--ccpd-root", type=str, required=True, help="raw_ccpd path")
    parser.add_argument("--out-root", type=str, default="data/det", help="output root")
    parser.add_argument("--train-list", type=str, required=True, help="path to all_train.txt")
    parser.add_argument("--test-list", type=str, required=True, help="path to all_test.txt")
    parser.add_argument(
        "--copy", action="store_true", help="copy images instead of symlink"
    )
    args = parser.parse_args()

    ccpd_root = Path(args.ccpd_root)
    out_root = Path(args.out_root)
    
    prepare(
        Path(args.train_list), 
        Path(args.test_list), 
        ccpd_root, 
        out_root, 
        copy=args.copy
    )


if __name__ == "__main__":
    main()

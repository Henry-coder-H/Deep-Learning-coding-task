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
from pipeline.utils import parse_ccpd_filename, warp_plate  # noqa: E402


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
            if e.errno == 17:
                dst.unlink(missing_ok=True)
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)


def prepare(
    train_txt: Path,
    val_txt: Path,
    ccpd_root: Path,
    out_root: Path,
    img_w: int,
    img_h: int,
    copy: bool,
) -> None:
    
    with open(train_txt, 'r') as f:
        train_lines = [line.strip() for line in f if line.strip()]
    with open(val_txt, 'r') as f:
        val_lines = [line.strip() for line in f if line.strip()]

    # We use "val" set here as validation for CRNN training
    # Later "val" set can also be used as test set
    
    labels = {"train": [], "val": []}
    
    for subset, lines in [("train", train_lines), ("val", val_lines)]:
        img_dir = out_root / "images" / subset
        img_dir.mkdir(parents=True, exist_ok=True)
        
        for line in tqdm(lines, desc=f"{subset}"):
            found_path = None
            for root_candidate in [ccpd_root / "CCPD2019", ccpd_root / "CCPD2020"]:
                p = root_candidate / line
                if p.exists():
                    found_path = p
                    break
            
            if not found_path:
                p = ccpd_root / line
                if p.exists():
                    found_path = p
            
            if not found_path:
                continue

            try:
                meta = parse_ccpd_filename(found_path.name)
            except Exception as exc:
                continue

            image = cv2.imread(str(found_path))
            if image is None:
                continue

            warped = warp_plate(image, meta["points"], img_w, img_h)
            
            # Save ONLY warped crop (no original image link)
            # Use original filename but with .jpg extension (warped is small)
            dst_path = img_dir / found_path.name
            cv2.imwrite(str(dst_path), warped)
            
            # Record label: path_to_crop \t text
            labels[subset].append(f"{dst_path}\t{meta['plate']}")

    for subset, lines in labels.items():
        label_file = out_root / f"{subset}_labels.txt"
        label_file.parent.mkdir(parents=True, exist_ok=True)
        label_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[INFO] {subset} samples: {len(lines)}, labels -> {label_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create CRNN training set from specified list files."
    )
    parser.add_argument("--ccpd-root", type=str, required=True, help="raw_ccpd path")
    parser.add_argument("--out-root", type=str, default="data/rec", help="output root")
    parser.add_argument("--train-list", type=str, required=True)
    parser.add_argument("--test-list", type=str, required=True) # acts as val/test
    parser.add_argument("--imgw", type=int, default=160)
    parser.add_argument("--imgh", type=int, default=32)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    ccpd_root = Path(args.ccpd_root)
    out_root = Path(args.out_root)

    prepare(
        Path(args.train_list),
        Path(args.test_list),
        ccpd_root,
        out_root,
        img_w=args.imgw,
        img_h=args.imgh,
        copy=args.copy,
    )


if __name__ == "__main__":
    main()

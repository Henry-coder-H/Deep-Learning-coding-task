import argparse
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from detection.inference import YOLODetector
from recognition.inference import CRNNRecognizer
from pipeline.utils import parse_ccpd_filename, warp_plate, CRNN_ALPHABET


def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def evaluate(args):
    # Load Models
    print("[INFO] Loading models...")
    detector = YOLODetector(args.det_weights, device=args.device)
    recognizer = CRNNRecognizer(args.rec_weights, device=args.device)
    
    # Memory Usage Estimation (Approximate)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        dummy = np.zeros((1080, 1920, 3), dtype=np.uint8)
        _ = detector.predict(dummy)
        _ = recognizer.predict(np.zeros((32, 160), dtype=np.float32))
        max_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[INFO] Estimated Peak GPU Memory: {max_mem:.2f} MB")
    
    # 1. Determine Image List
    # If args.img_list is provided (e.g. all_hardtest.txt), read it
    # Else if args.img_root provided, scan it
    
    images = []
    ccpd_root = Path(args.img_root) if args.img_root else None
    
    if args.img_list:
        list_path = Path(args.img_list)
        if not list_path.exists():
            raise FileNotFoundError(f"{list_path} not found")
        with open(list_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Resolve paths
        # Assuming lines are relative to ccpd_root or are absolute?
        # Usually relative like "ccpd_rotate/xxx.jpg"
        for line in lines:
            # Try finding it
            found = False
            if ccpd_root:
                candidates = [
                    ccpd_root / line,
                    ccpd_root / "CCPD2019" / line,
                    ccpd_root / "CCPD2020" / line,
                    ccpd_root / "raw_ccpd" / "CCPD2019" / line # Fallback structure guess
                ]
                for cand in candidates:
                    if cand.exists():
                        images.append(cand)
                        found = True
                        break
            if not found:
                 # Try absolute if user gave absolute path
                 if Path(line).exists():
                     images.append(Path(line))
    
    elif ccpd_root:
        images = list(ccpd_root.rglob("*.jpg"))
    
    if args.max_samples and len(images) > args.max_samples:
        images = images[:args.max_samples]
        
    print(f"[INFO] Evaluating on {len(images)} images...")
    
    if not images:
        print("[WARN] No images found to evaluate.")
        return

    # Metrics
    total_imgs = 0
    correct_full = 0
    total_chars = 0
    correct_chars = 0
    
    # Robustness
    tilt_correct = 0
    tilt_total = 0
    
    # Latency
    times = []
    
    for img_path in tqdm(images):
        try:
            meta = parse_ccpd_filename(img_path.name)
            gt_plate = meta['plate']
            tilt = meta['tilt']
        except Exception:
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        t0 = time.time()
        
        # 1. Detection
        dets = detector.predict(img)
        
        if not dets:
            pred_plate = ""
        else:
            det = max(dets, key=lambda x: x['conf'])
            
            # 2. Warp
            if args.no_warp:
                x1, y1, x2, y2 = map(int, det['bbox'])
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    crop = np.zeros((32, 160, 3), dtype=np.uint8)
            else:
                crop = warp_plate(img, det['points'], 160, 32)
                
            # 3. Recognition
            pred_plate = recognizer.predict(crop)
            
        t1 = time.time()
        times.append((t1 - t0) * 1000)
        
        # Metrics Calculation
        total_imgs += 1
        if pred_plate == gt_plate:
            correct_full += 1
            if tilt > 30:
                tilt_correct += 1
                
        if tilt > 30:
            tilt_total += 1
            
        dist = edit_distance(pred_plate, gt_plate)
        match_len = max(0, len(gt_plate) - dist)
        correct_chars += match_len
        total_chars += len(gt_plate)
        
    # Summary
    acc_full = correct_full / total_imgs if total_imgs else 0
    acc_char = correct_chars / total_chars if total_chars else 0
    avg_latency = np.mean(times) if times else 0
    fps = 1000 / avg_latency if avg_latency > 0 else 0
    
    acc_tilt = tilt_correct / tilt_total if tilt_total else 0
    
    print("\n" + "="*40)
    print("           Evaluation Results           ")
    print("="*40)
    print(f"Dataset Size:          {total_imgs}")
    print(f"Accuracy (Full Match): {acc_full*100:.2f}% (Target > 95%)")
    print(f"Accuracy (Char Level): {acc_char*100:.2f}%")
    print(f"Latency (ms/img):      {avg_latency:.2f} ms")
    print(f"FPS:                   {fps:.1f} (Target > 30)")
    print(f"Robustness (>30 deg):  {acc_tilt*100:.2f}% (Total >30 deg samples: {tilt_total})")
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-weights", type=str, required=True)
    parser.add_argument("--rec-weights", type=str, required=True)
    parser.add_argument("--img-root", type=str, default="raw_ccpd", help="Root for relative paths")
    parser.add_argument("--img-list", type=str, help="Path to txt list file (optional)")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--no-warp", action="store_true", help="Disable perspective warp")
    args = parser.parse_args()
    
    evaluate(args)

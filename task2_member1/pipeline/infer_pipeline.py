import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from detection.inference import YOLODetector
from recognition.inference import CRNNRecognizer
from pipeline.utils import warp_plate


def draw_results(image, results, fps=None):
    vis = image.copy()
    for res in results:
        # Draw bbox
        box = np.array(res['bbox'], dtype=int)
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        # Draw points
        pts = np.array(res['points'], dtype=int)
        for pt in pts:
            cv2.circle(vis, tuple(pt), 4, (0, 0, 255), -1)
            
        # Draw text
        text = res.get('text', '')
        cv2.putText(vis, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 255, 0), 2)
                    
    if fps:
        cv2.putText(vis, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 255, 255), 3)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det-weights", type=str, required=True)
    parser.add_argument("--rec-weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--out-dir", type=str, default="runs/predict")
    args = parser.parse_args()
    
    # Initialize models
    detector = YOLODetector(args.det_weights, device=args.device)
    recognizer = CRNNRecognizer(args.rec_weights, device=args.device)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle video or image
    source = args.source
    is_video = source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    if is_video:
        cap = cv2.VideoCapture(source)
        fps_hist = []
        
        if args.save_vis:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = out_dir / f"{Path(source).stem}_res.mp4"
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
                
            # 1. Detect
            dets = detector.predict(frame)
            
            # 2. Warp & Recognize
            crops = []
            valid_dets = []
            for det in dets:
                pts = det['points']
                # Warp
                warped = warp_plate(frame, pts, 160, 32)
                crops.append(warped)
                valid_dets.append(det)
                
            if crops:
                texts = recognizer.predict_batch(crops)
                for det, text in zip(valid_dets, texts):
                    det['text'] = text
            
            t1 = time.time()
            dt = t1 - t0
            curr_fps = 1.0 / dt if dt > 0 else 0
            fps_hist.append(curr_fps)
            avg_fps = np.mean(fps_hist[-50:])
            
            # Vis
            if args.save_vis:
                vis = draw_results(frame, valid_dets, avg_fps)
                writer.write(vis)
            
            print(f"\rFPS: {avg_fps:.1f} | Plates: {len(valid_dets)}", end="")
            
        cap.release()
        if args.save_vis:
            writer.release()
            print(f"\nSaved to {out_path}")
            
    else:
        # Image
        img = cv2.imread(source)
        if img is None:
            raise ValueError(f"Could not read {source}")
            
        t0 = time.time()
        dets = detector.predict(img)
        
        crops = []
        for det in dets:
            pts = det['points']
            warped = warp_plate(img, pts, 160, 32)
            crops.append(warped)
            
        if crops:
            texts = recognizer.predict_batch(crops)
            for det, text in zip(dets, texts):
                det['text'] = text
        
        print(f"Inference time: {(time.time() - t0)*1000:.1f}ms")
        for det in dets:
            print(f"Plate: {det.get('text', 'N/A')} Conf: {det['conf']:.2f}")
            
        if args.save_vis:
            vis = draw_results(img, dets)
            out_path = out_dir / f"{Path(source).name}"
            cv2.imwrite(str(out_path), vis)
            print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()




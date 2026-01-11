from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, model_path: str, device: str = "0", conf: float = 0.25):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def predict(self, image: np.ndarray) -> List[dict]:
        """
        Return list of dicts: {'bbox': [x1, y1, x2, y2], 'points': [[x1,y1], ...], 'conf': float}
        """
        results = self.model.predict(image, device=self.device, conf=self.conf, verbose=False)
        output = []
        
        for res in results:
            if not res.boxes:
                continue
                
            boxes = res.boxes.xyxy.cpu().numpy()  # [N, 4]
            # keypoints exist?
            if res.keypoints is not None:
                # [N, 4, 2] or [N, 4, 3]
                kpts = res.keypoints.xy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    pts = kpts[i]  # [[x,y], [x,y], [x,y], [x,y]]
                    output.append({
                        "bbox": box.tolist(),
                        "points": pts.tolist(),
                        "conf": float(confs[i])
                    })
        return output




import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from pipeline.utils import CRNN_ALPHABET
from recognition.model import CRNN
from recognition.utils import CTCLabelConverter


class CRNNRecognizer:
    def __init__(self, model_path: str, device: str = "0", img_w: int = 160, img_h: int = 32):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_w = img_w
        self.img_h = img_h
        
        self.converter = CTCLabelConverter(CRNN_ALPHABET)
        num_classes = len(CRNN_ALPHABET) + 1
        
        # Use ResNetCRNN
        from recognition.model import ResNetCRNN
        self.model = ResNetCRNN(num_classes=num_classes, img_h=img_h, nc=1)
        # Load weights
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: np.ndarray) -> str:
        """
        Predict single image (cropped plate).
        Image should be BGR or Grayscale.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Preprocess
        gray = cv2.resize(gray, (self.img_w, self.img_h))
        gray = gray.astype(np.float32) / 255.0
        gray = (gray - 0.5) / 0.5
        
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            preds = self.model(tensor) # [T, 1, C]
            
            T, B, _ = preds.shape
            pred_lens = torch.full((B,), T, dtype=torch.long, device=self.device)
            
            texts = self.converter.decode(preds, pred_lens)
            return texts[0]

    def predict_batch(self, images: List[np.ndarray]) -> List[str]:
        if not images:
            return []
            
        tensors = []
        for img in images:
            if img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            gray = cv2.resize(gray, (self.img_w, self.img_h))
            gray = gray.astype(np.float32) / 255.0
            gray = (gray - 0.5) / 0.5
            tensors.append(torch.from_numpy(gray).unsqueeze(0))
            
        batch = torch.stack(tensors) # [B, 1, H, W]
        batch = batch.to(self.device)
        
        with torch.no_grad():
            preds = self.model(batch) # [T, B, C]
            T, B, _ = preds.shape
            pred_lens = torch.full((B,), T, dtype=torch.long, device=self.device)
            texts = self.converter.decode(preds, pred_lens)
            return texts




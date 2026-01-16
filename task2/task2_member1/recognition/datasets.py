from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LPRDataset(Dataset):
    def __init__(self, label_file: Path, img_w: int = 160, img_h: int = 32):
        self.label_file = Path(label_file)
        self.img_w = img_w
        self.img_h = img_h
        self.samples: List[Tuple[Path, str]] = []
        self._load()

    def _load(self) -> None:
        if not self.label_file.exists():
            raise FileNotFoundError(self.label_file)
        lines = self.label_file.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            if "\t" not in line:
                continue
            path_str, text = line.split("\t", maxsplit=1)
            self.samples.append((Path(path_str), text.strip()))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, text = self.samples[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        tensor = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        return tensor, text


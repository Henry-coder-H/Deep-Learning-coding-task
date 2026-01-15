"""
车牌识别模块 - 基于 YOLOv8-Pose + ResNet-CRNN
复用 task2_member1 的实现
"""
import sys
from pathlib import Path
from typing import List, Tuple, Sequence, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# =========================
# 字符字典 (来自 task2_member1/pipeline/utils.py)
# =========================
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
]

CRNN_ALPHABET = PROVINCES + list("ABCDEFGHJKLMNPQRSTUVWXYZ") + list("0123456789")

# 后处理映射：修正因训练数据ADS顺序错误导致的字符错位
# 训练时用的错误ADS（数字在前） vs 正确ADS（字母在前）
_WRONG_ADS = list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ")
_CORRECT_ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789")
PLATE_FIX_MAP = {_WRONG_ADS[i]: _CORRECT_ADS[i] for i in range(len(_WRONG_ADS))}


def fix_plate_text(plate: str) -> str:
    """修正模型输出的车牌文本（第3-7位字符映射）"""
    if len(plate) <= 2:
        return plate
    # 前2位（省份+字母）不变，第3-7位做映射修正
    fixed = plate[:2]
    for c in plate[2:]:
        fixed += PLATE_FIX_MAP.get(c, c)
    return fixed


# =========================
# 工具函数 (来自 task2_member1/pipeline/utils.py)
# =========================
def order_points(points: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Order 4 points as tl, tr, br, bl."""
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_plate(
    image: np.ndarray,
    points: Sequence[Tuple[float, float]],
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Perspective warp to (out_w, out_h) using 4 source points."""
    src = order_points(points)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


# =========================
# CTC Label Converter (与 task2_member1 一致的实现)
# =========================
class CTCLabelConverter:
    """Encode/Decode between text-label and CTC tensor."""

    def __init__(self, alphabet):
        self.alphabet = list(alphabet)
        self.blank = len(self.alphabet)  # blank 在最后
        self.char_to_idx = {ch: i for i, ch in enumerate(self.alphabet)}

    def encode(self, texts):
        lengths = [len(t) for t in texts]
        total_len = sum(lengths)
        targets = torch.full((total_len,), self.blank, dtype=torch.long)
        idx = 0
        for text in texts:
            for ch in text:
                targets[idx] = self.char_to_idx.get(ch, self.blank - 1)
                idx += 1
        return targets, torch.tensor(lengths, dtype=torch.long)

    def decode(self, preds, pred_lens):
        """Greedy decode. preds: [T, B, C], pred_lens: [B]."""
        max_idx = preds.argmax(2)  # [T, B]
        max_idx = max_idx.permute(1, 0)  # [B, T]
        texts = []
        for idxs, plen in zip(max_idx, pred_lens):
            seq = idxs[: plen].tolist()
            # Collapse repeats and blanks
            prev = self.blank
            chars = []
            for s in seq:
                if s != prev and s != self.blank and s < len(self.alphabet):
                    chars.append(self.alphabet[s])
                prev = s
            texts.append("".join(chars))
        return texts


# =========================
# ResNet-CRNN 模型 (来自 task2_member1/recognition/model.py)
# =========================
class ResNetCRNN(nn.Module):
    def __init__(self, num_classes: int, img_h: int = 32, nc: int = 1, nh: int = 256):
        super().__init__()
        
        input_channel = nc
        hidden_size = nh
        
        resnet = models.resnet18(weights=None)
        
        if input_channel != 3:
            resnet.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        
        self.layer3 = resnet.layer3
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)
        
        self.layer4 = resnet.layer4
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)
        
        self.layer2[0].conv1.stride = (2, 1)
        self.layer2[0].downsample[0].stride = (2, 1)
        
        self.lstm = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, batch_first=False)
        self.embedding = nn.Linear(hidden_size * 2, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        nn.init.normal_(self.embedding.weight, 0, 0.01)
        nn.init.constant_(self.embedding.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        assert x.size(2) == 1, f"Expected H=1, got {x.size(2)}"
        x = x.squeeze(2)
        
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.embedding(x)
        return x


# =========================
# YOLOv8-Pose 检测器 (来自 task2_member1/detection/inference.py)
# =========================
class YOLODetector:
    def __init__(self, model_path: str, device: str = "0", conf: float = 0.25):
        from ultralytics import YOLO
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
                
            boxes = res.boxes.xyxy.cpu().numpy()
            if res.keypoints is not None:
                kpts = res.keypoints.xy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    pts = kpts[i]
                    output.append({
                        "bbox": box.tolist(),
                        "points": pts.tolist(),
                        "conf": float(confs[i])
                    })
        return output


# =========================
# CRNN 识别器 (来自 task2_member1/recognition/inference.py)
# =========================
class CRNNRecognizer:
    def __init__(self, model_path: str, device: str = "cuda:0", img_w: int = 160, img_h: int = 32):
        if device in ["0", "cuda", "gpu"]:
            device = "cuda:0"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.img_w = img_w
        self.img_h = img_h
        
        self.converter = CTCLabelConverter(CRNN_ALPHABET)
        num_classes = len(CRNN_ALPHABET) + 1
        
        self.model = ResNetCRNN(num_classes=num_classes, img_h=img_h, nc=1)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: np.ndarray) -> str:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        gray = cv2.resize(gray, (self.img_w, self.img_h))
        gray = gray.astype(np.float32) / 255.0
        gray = (gray - 0.5) / 0.5
        
        tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            preds = self.model(tensor)
            T, B, _ = preds.shape
            pred_lens = torch.full((B,), T, dtype=torch.long, device=self.device)
            texts = self.converter.decode(preds, pred_lens)
            return fix_plate_text(texts[0])

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
            
        batch = torch.stack(tensors)
        batch = batch.to(self.device)
        
        with torch.no_grad():
            preds = self.model(batch)
            T, B, _ = preds.shape
            pred_lens = torch.full((B,), T, dtype=torch.long, device=self.device)
            texts = self.converter.decode(preds, pred_lens)
            return [fix_plate_text(t) for t in texts]


# =========================
# 车牌识别 Pipeline (整合检测+识别)
# =========================
class PlateRecognizer:
    """车牌识别完整流水线"""
    
    def __init__(self, det_weights: str, rec_weights: str, device: str = "cuda:0"):
        """
        初始化车牌识别器
        
        Args:
            det_weights: YOLOv8-Pose 检测模型权重路径
            rec_weights: CRNN 识别模型权重路径
            device: 推理设备 ("cuda:0" for GPU, "cpu" for CPU)
        """
        # YOLO 可以接受 "0" 或 "cuda:0"，这里统一处理
        yolo_device = "0" if device.startswith("cuda") else device
        self.detector = YOLODetector(det_weights, device=yolo_device)
        self.recognizer = CRNNRecognizer(rec_weights, device=device)
        
    def recognize_image(self, image: np.ndarray) -> List[Dict]:
        """
        对单张图片进行车牌识别
        
        Args:
            image: BGR格式的图像
            
        Returns:
            List[Dict]: 每个检测结果包含 bbox, points, conf, text
        """
        dets = self.detector.predict(image)
        
        crops = []
        valid_dets = []
        for det in dets:
            pts = det['points']
            try:
                warped = warp_plate(image, pts, 160, 32)
                crops.append(warped)
                valid_dets.append(det)
            except Exception:
                continue
                
        if crops:
            texts = self.recognizer.predict_batch(crops)
            for det, text in zip(valid_dets, texts):
                det['text'] = text
                
        return valid_dets
    
    def recognize_frame(self, frame: np.ndarray) -> Tuple[List[Dict], List[np.ndarray]]:
        """
        对视频帧进行车牌识别，同时返回裁剪的车牌图像
        
        Args:
            frame: BGR格式的视频帧
            
        Returns:
            Tuple[List[Dict], List[np.ndarray]]: (检测结果, 裁剪的车牌图像列表)
        """
        dets = self.detector.predict(frame)
        
        crops = []
        valid_dets = []
        for det in dets:
            pts = det['points']
            try:
                warped = warp_plate(frame, pts, 160, 32)
                crops.append(warped)
                valid_dets.append(det)
            except Exception:
                continue
                
        if crops:
            texts = self.recognizer.predict_batch(crops)
            for det, text in zip(valid_dets, texts):
                det['text'] = text
                
        return valid_dets, crops

import cv2
import numpy as np
from ultralytics import YOLO
import sys

# 定义 Task1 中的颜色配置
COLORS = {
    'Bus': (0, 128, 255), 'Microbus': (0, 255, 255), 'Minivan': (255, 0, 255),
    'Sedan': (0, 255, 0), 'SUV': (255, 0, 0), 'Truck': (0, 0, 255)
}
DEFAULT_COLOR = (255, 255, 255)

class VehicleTypeClassifier:
    def __init__(self, model_path=None):
        """
        初始化车型识别模型
        """
        if model_path:
            print(f"👉 [车型识别] 加载 YOLO 模型: {model_path}")
            try:
                self.model = YOLO(model_path)
                self.class_names = self.model.names
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                self.model = None
        else:
            self.model = None
            print("⚠️ 未指定模型路径，车型识别功能将不可用")

    def get_color(self, class_name):
        """获取对应车型的颜色 (BGR)"""
        return COLORS.get(class_name, DEFAULT_COLOR)

    def predict(self, image: np.ndarray, conf=0.25):
        """
        图片预测模式
        Returns:
            list: 检测结果列表 [{'bbox': [x1,y1,x2,y2], 'class_name': str, 'conf': float}]
        """
        if self.model is None:
            return []

        results = self.model.predict(image, conf=conf, verbose=False)[0]
        detections = []

        if results.boxes:
            boxes = results.boxes.xyxy.cpu().numpy()
            clses = results.boxes.cls.int().cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                class_name = self.class_names[clses[i]]
                if class_name == 'Truck':
                    continue
                detections.append({
                    'bbox': box,
                    'class_name': class_name,
                    'conf': confs[i],
                    'track_id': -1 # 图片模式无追踪ID
                })
        return detections

    def track(self, image: np.ndarray, conf=0.25):
        """
        视频追踪模式
        Returns:
            list: 检测结果列表 (含 track_id)
        """
        if self.model is None:
            return []

        # 使用 track 模式，开启 persist=True 以保持ID
        results = self.model.track(image, persist=True, conf=conf, verbose=False)[0]
        detections = []

        if results.boxes:
            boxes = results.boxes.xyxy.cpu().numpy()
            clses = results.boxes.cls.int().cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            # 获取 ID，如果没有ID（第一帧可能）则设为 -1
            ids = results.boxes.id.int().cpu().numpy() if results.boxes.id is not None else [-1] * len(boxes)

            for i, box in enumerate(boxes):
                class_name = self.class_names[clses[i]]
                if class_name == 'Truck':
                    continue
                detections.append({
                    'bbox': box,
                    'class_name': class_name,
                    'conf': confs[i],
                    'track_id': ids[i]
                })
        return detections

    def get_type_from_coco_id(self, coco_id):
        """兼容旧接口，暂时保留"""
        return "Unknown"
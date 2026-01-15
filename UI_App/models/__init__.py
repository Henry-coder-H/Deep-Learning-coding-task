"""
模型模块
"""
from .plate_recognizer import PlateRecognizer, YOLODetector, CRNNRecognizer
from .speed_estimator import VehicleSpeedEstimator
from .vehicle_classifier import VehicleTypeClassifier

__all__ = [
    'PlateRecognizer',
    'YOLODetector', 
    'CRNNRecognizer',
    'VehicleSpeedEstimator',
    'VehicleTypeClassifier'
]

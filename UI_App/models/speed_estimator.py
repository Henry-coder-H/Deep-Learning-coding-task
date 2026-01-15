"""
车速识别模块 - 基于 YOLOv11 + 单应性变换 + 卡尔曼滤波
复用 task_3 的实现
"""
import cv2
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import json
import warnings

warnings.filterwarnings('ignore')


class VehicleSpeedEstimator:
    """
    车速估计器
    使用单应性变换将像素坐标转换为真实世界坐标，
    结合卡尔曼滤波和多帧速度平滑算法计算车辆实时速度
    """
    
    def __init__(self, fps: float = 30.0, vehicle_model_path: str = "yolov11l.pt"):
        """
        初始化车速估计器
        
        Args:
            fps: 视频帧率
            vehicle_model_path: YOLO车辆检测模型路径
        """
        from ultralytics import YOLO
        
        self.fps = fps
        self.frame_width = 0
        self.frame_height = 0
        
        # 加载YOLO模型
        print("正在加载YOLOv11l模型...")
        self.model = YOLO(vehicle_model_path)
        
        # 车辆类别ID（COCO数据集）
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
        # 标定相关
        self.homography_matrix = None
        self.calibrated = False
        self.calibration_error = None
        
        # 追踪和速度计算
        self.tracks = defaultdict(list)
        self.speeds = defaultdict(lambda: deque(maxlen=40))
        self.prev_detections = {}
        self.next_track_id = 0
        self.kalman_filters = {}
        
        # 可视化颜色
        self.colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(100)]
        
    def set_frame_size(self, width: int, height: int):
        """设置帧尺寸"""
        self.frame_width = width
        self.frame_height = height
        
    def reset(self):
        """重置追踪状态（用于处理新视频）"""
        self.tracks = defaultdict(list)
        self.speeds = defaultdict(lambda: deque(maxlen=40))
        self.prev_detections = {}
        self.next_track_id = 0
        self.kalman_filters = {}
        
    def calibrate_from_points(self, pixel_points: List[Tuple[float, float]], 
                               world_points: List[Tuple[float, float]]) -> bool:
        """
        从已知点对进行标定
        
        Args:
            pixel_points: 像素坐标点列表 [(x, y), ...]
            world_points: 对应的世界坐标点列表 [(x, y), ...] 单位: 米
            
        Returns:
            bool: 标定是否成功
        """
        if len(pixel_points) < 4 or len(pixel_points) != len(world_points):
            print("错误：需要至少4个匹配点对")
            return False
            
        pixel_pts = np.float32(pixel_points)
        world_pts = np.float32(world_points)
        
        self.homography_matrix, mask = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC, 5.0)
        
        if self.homography_matrix is None:
            print("错误：无法计算单应性矩阵")
            return False
            
        self.calibrated = True
        self.calibration_error = self._calculate_calibration_error(pixel_pts, world_pts)
        
        print(f"标定完成！标定点数量: {len(pixel_points)}, 误差: {self.calibration_error:.4f} 像素")
        return True
        
    def _calculate_calibration_error(self, pixel_points: np.ndarray, 
                                      world_points: np.ndarray) -> float:
        """计算标定误差"""
        world_to_pixel_matrix = np.linalg.inv(self.homography_matrix)
        
        errors = []
        for world_point, pixel_point in zip(world_points, pixel_points):
            world_pt = np.array([[world_point[0], world_point[1]]], dtype=np.float32)
            predicted_pixel = cv2.perspectiveTransform(world_pt.reshape(-1, 1, 2), world_to_pixel_matrix)
            
            predicted_x = predicted_pixel[0][0][0]
            predicted_y = predicted_pixel[0][0][1]
            
            error = np.sqrt((predicted_x - pixel_point[0]) ** 2 + (predicted_y - pixel_point[1]) ** 2)
            errors.append(error)
            
        return np.mean(errors)
        
    def validate_lane_width(self, point1: Tuple[float, float], 
                            point2: Tuple[float, float]) -> Tuple[float, str]:
        """
        验证车道宽度标定准确性
        
        Args:
            point1: 车道左边缘像素坐标
            point2: 车道右边缘像素坐标
            
        Returns:
            Tuple[float, str]: (计算的宽度(米), 评估信息)
        """
        if not self.calibrated:
            return 0, "未标定"
            
        world_left = self.pixel_to_world(point1)
        world_right = self.pixel_to_world(point2)
        
        # 计算距离 (世界坐标单位已经是米，无需除以100)
        distance_m = np.sqrt(
            (world_right[0] - world_left[0]) ** 2 + 
            (world_right[1] - world_left[1]) ** 2
        )
        # distance_m = distance_cm / 100.0  <-- 已移除
        
        # 标准车道宽度3.75米
        error = abs(distance_m - 3.75)
        error_percent = error / 3.75 * 100
        
        if error <= 0.3:
            status = f"✅ 良好 (误差 {error:.2f}m, {error_percent:.1f}%)"
        elif error <= 0.5:
            status = f"⚠️ 可接受 (误差 {error:.2f}m, {error_percent:.1f}%)"
        else:
            status = f"❌ 较大误差 (误差 {error:.2f}m, {error_percent:.1f}%)"
            
        return distance_m, status
        
    def pixel_to_world(self, pixel_point: Tuple[float, float]) -> Tuple[float, float]:
        """
        像素坐标转换为世界坐标
        
        Args:
            pixel_point: (x, y) 像素坐标
            
        Returns:
            (world_x, world_y) 世界坐标（米）
        """
        if not self.calibrated:
            raise ValueError("请先进行相机标定！")
            
        pixel_point = np.array([pixel_point], dtype=np.float32)
        
        try:
            world_points = cv2.perspectiveTransform(pixel_point.reshape(-1, 1, 2), self.homography_matrix)
            world_x = float(world_points[0][0][0])
            world_y = float(world_points[0][0][1])
            return world_x, world_y
        except Exception as e:
            print(f"坐标转换错误: {e}")
            return 0, 0
            
    def init_kalman_filter(self, track_id: int, initial_position: Tuple[float, float]):
        """初始化卡尔曼滤波器"""
        kalman = cv2.KalmanFilter(4, 2)
        
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        kalman.errorCovPost = np.eye(4, dtype=np.float32)
        kalman.statePost = np.array([initial_position[0], initial_position[1], 0, 0], dtype=np.float32)
        
        self.kalman_filters[track_id] = kalman
        
    def update_kalman_filter(self, track_id: int, 
                              measurement: Tuple[float, float]) -> Tuple[float, float]:
        """更新卡尔曼滤波器并返回滤波后的位置"""
        if track_id not in self.kalman_filters:
            self.init_kalman_filter(track_id, measurement)
            return measurement
            
        kalman = self.kalman_filters[track_id]
        kalman.predict()
        
        z = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
        kalman.correct(z)
        
        filtered_state = kalman.statePost
        return float(filtered_state[0]), float(filtered_state[1])
        
    def calculate_iou(self, box1, box2) -> float:
        """计算两个框的IoU"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
        
    def simple_tracking(self, detections: List, iou_threshold: float = 0.3) -> List:
        """简单IoU追踪算法"""
        current_tracks = {}
        
        if not self.prev_detections:
            for i, det in enumerate(detections):
                track_id = self.next_track_id
                current_tracks[track_id] = det + (track_id,)
                self.next_track_id += 1
            self.prev_detections = current_tracks.copy()
            return list(current_tracks.values())
            
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.prev_detections.keys())
        matched_pairs = []
        
        for i in unmatched_detections[:]:
            best_iou = 0
            best_track_id = None
            
            for track_id in unmatched_tracks:
                iou = self.calculate_iou(detections[i][:4], self.prev_detections[track_id][:4])
                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
                    
            if best_track_id is not None:
                matched_pairs.append((i, best_track_id))
                unmatched_detections.remove(i)
                unmatched_tracks.remove(best_track_id)
                
        for det_idx, track_id in matched_pairs:
            current_tracks[track_id] = detections[det_idx] + (track_id,)
            
        for det_idx in unmatched_detections:
            track_id = self.next_track_id
            current_tracks[track_id] = detections[det_idx] + (track_id,)
            self.next_track_id += 1
            
        self.prev_detections = {track_id: det[:6] for track_id, det in current_tracks.items()}
        
        return list(current_tracks.values())
        
    def calculate_speed(self, track_id: int, current_position: Tuple[float, float], 
                        frame_idx: int) -> float:
        """计算速度（km/h）"""
        history = self.tracks.get(track_id, [])
        
        if len(history) < 2:
            return 0.0
            
        # 多帧线性回归
        N = min(5, len(history))
        recent_history = history[-N:]
        
        frames = [h[0] for h in recent_history]
        x_positions = [h[1] for h in recent_history]
        y_positions = [h[2] for h in recent_history]
        
        total_time = (frames[-1] - frames[0]) / self.fps
        
        if total_time > 0:
            # 总位移 (米)
            total_distance = np.sqrt(
                (x_positions[-1] - x_positions[0]) ** 2 +
                (y_positions[-1] - y_positions[0]) ** 2
            )
            
            # 速度 (m/s -> km/h)
            speed_ms = total_distance / total_time
            speed_kmh = speed_ms * 3.6  # m/s to km/h (系数已修改)
            
            # 两帧速度作为补充
            if len(history) >= 2:
                dt2 = (history[-1][0] - history[-2][0]) / self.fps
                distance2 = np.sqrt(
                    (history[-1][1] - history[-2][1]) ** 2 +
                    (history[-1][2] - history[-2][2]) ** 2
                )
                speed_kmh2 = (distance2 / dt2) * 3.6 if dt2 > 0 else 0
                
                # 加权平均
                final_speed = speed_kmh * 0.7 + speed_kmh2 * 0.3
            else:
                final_speed = speed_kmh
        else:
            return 0.0
            
        # 合理性检查
        if final_speed < 0 or final_speed > 200:
            if len(self.speeds[track_id]) > 0:
                final_speed = np.mean(list(self.speeds[track_id])[-5:])
                
        self.speeds[track_id].append(final_speed)
        
        # 滑动平均平滑
        if len(self.speeds[track_id]) > 0:
            return np.mean(self.speeds[track_id])
        return final_speed
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict]:
        """
        处理单帧
        
        Args:
            frame: 输入帧
            frame_idx: 帧索引
            
        Returns:
            Tuple[np.ndarray, Dict]: (处理后的帧, 速度信息字典)
        """
        if frame is None:
            return None, {}
            
        if self.frame_width == 0:
            self.frame_height, self.frame_width = frame.shape[:2]
            
        # YOLO检测
        results = self.model(frame, verbose=False, save=False, save_txt=False, save_conf=False)[0]
        
        vehicle_detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    vehicle_detections.append((x1, y1, x2, y2, conf, cls_id))
                    
        # 追踪
        tracked_detections = self.simple_tracking(vehicle_detections)
        
        speeds_info = {}
        processed_frame = frame.copy()
        
        for det in tracked_detections:
            x1, y1, x2, y2, conf, cls_id, track_id = det
            
            # 底部中心点
            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2
            
            if self.calibrated:
                try:
                    world_x, world_y = self.pixel_to_world((bottom_center_x, bottom_center_y))
                    filtered_x, filtered_y = self.update_kalman_filter(track_id, (world_x, world_y))
                    speed_kmh = self.calculate_speed(track_id, (filtered_x, filtered_y), frame_idx)
                    
                    self.tracks[track_id].append((frame_idx, filtered_x, filtered_y, speed_kmh))
                    
                    if len(self.tracks[track_id]) > 100:
                        self.tracks[track_id].pop(0)
                        
                    speeds_info[track_id] = {
                        'speed': speed_kmh,
                        'position': (filtered_x, filtered_y),
                        'pixel_position': (bottom_center_x, bottom_center_y),
                        'class_id': cls_id,
                        'bbox': (x1, y1, x2, y2)
                    }
                    
                except Exception as e:
                    print(f"处理车辆{track_id}时出错: {e}")
                    continue
            else:
                # 未标定时仍然输出检测框但不计算速度
                speeds_info[track_id] = {
                    'speed': 0,
                    'position': (0, 0),
                    'pixel_position': (bottom_center_x, bottom_center_y),
                    'class_id': cls_id,
                    'bbox': (x1, y1, x2, y2)
                }
                    
        return processed_frame, speeds_info
        
    def get_first_frame(self, video_path: str) -> Optional[np.ndarray]:
        """获取视频第一帧用于标定"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.frame_height, self.frame_width = frame.shape[:2]
            return frame
        return None
        
    def save_calibration(self, filepath: str):
        """保存标定数据"""
        if not self.calibrated:
            return
            
        data = {
            'homography_matrix': self.homography_matrix.tolist(),
            'calibration_error': self.calibration_error,
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'fps': self.fps
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    def load_calibration(self, filepath: str) -> bool:
        """加载标定数据"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.homography_matrix = np.array(data['homography_matrix'])
            self.calibration_error = data['calibration_error']
            self.calibrated = True
            return True
        except Exception as e:
            print(f"加载标定失败: {e}")
            return False
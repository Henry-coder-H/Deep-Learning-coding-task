import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import warnings
import json
import os

warnings.filterwarnings('ignore')


class VehicleSpeedEstimator:
    def __init__(self, video_path, fps=30, reference_points=None):
        """
        初始化车速估计器

        Args:
            video_path: 视频路径
            fps: 视频帧率（如果不知道，设为None会自动检测）
            reference_points: 标定点坐标，格式为[(pixel_x, pixel_y, world_x, world_y), ...]
        """
        # 初始化视频
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频属性
        self.fps = fps if fps else self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30  # 默认值
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 加载YOLOv11l模型
        print("正在加载YOLOv11l模型...")
        self.model = YOLO('yolov11l.pt')

        # 车辆类别ID（COCO数据集中car=2, truck=7, bus=5等）
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

        # 标定相关
        self.reference_points = reference_points
        self.homography_matrix = None
        self.calibrated = False
        self.calibration_error = None  # 标定误差

        # 存储车辆轨迹和速度
        self.tracks = defaultdict(list)  # {track_id: [(frame_idx, world_x, world_y, speed), ...]}
        self.speeds = defaultdict(lambda: deque(maxlen=40))  # 存储最近40个速度值用于平滑

        # 追踪相关
        self.prev_detections = {}  # 上一帧的检测结果 {track_id: [x1, y1, x2, y2, conf, cls_id]}
        self.next_track_id = 0

        # 卡尔曼滤波器字典
        self.kalman_filters = {}

        # 可视化参数
        self.colors = []
        for i in range(100):
            self.colors.append(tuple(np.random.randint(0, 255, 3).tolist()))

    def calibrate_camera(self, frame=None, min_points=6, max_points=12):
        """
        手动标定相机（如果没有提供标定点）

        Args:
            frame: 用于标定的帧
            min_points: 最小标定点数（推荐至少6个）
            max_points: 最大标定点数
        """
        if frame is None:
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError("无法读取视频帧进行标定")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到第一帧

        # 如果提供了标定点，直接计算单应性矩阵
        if self.reference_points:
            pixel_points = np.float32([(p[0], p[1]) for p in self.reference_points])
            world_points = np.float32([(p[2], p[3]) for p in self.reference_points])

            # 检查点数
            if len(pixel_points) < min_points:
                print(f"警告：提供的标定点只有{len(pixel_points)}个，建议至少{min_points}个")

            self.homography_matrix, mask = cv2.findHomography(pixel_points, world_points, cv2.RANSAC, 5.0)
            self.calibrated = True

            # 计算标定误差
            self.calibration_error = self.calculate_calibration_error(pixel_points, world_points)

            print("标定完成！")
            print(f"标定点数量: {len(pixel_points)}")
            print(f"单应性矩阵:")
            print(self.homography_matrix)
            print(f"标定误差: {self.calibration_error:.4f} 像素")

            # 保存标定结果
            self.save_calibration_data(pixel_points, world_points)
            return

        # 交互式标定 - 增加点数
        print("\n=== 相机标定 ===")
        print(f"请在图像上点击至少{min_points}个地面参考点（按顺序）")
        print("推荐选择：车道线交叉点、路面标记、已知尺寸物体等")
        print("注意：应该选择不同深度的点（近处和远处都要有）")
        print("按'c'完成标定，按'q'取消")

        ref_points = []
        temp_frame = frame.copy()

        def click_event(event, x, y, flags, param):
            nonlocal ref_points, temp_frame
            if event == cv2.EVENT_LBUTTONDOWN:
                ref_points.append((x, y))
                cv2.circle(temp_frame, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(temp_frame, f'P{len(ref_points)}', (x + 10, y + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Calibration', temp_frame)

        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Calibration', 1200, 800)
        cv2.imshow('Calibration', temp_frame)
        cv2.setMouseCallback('Calibration', click_event)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if len(ref_points) >= min_points:
                    break
                else:
                    print(f"至少需要{min_points}个标定点！当前只有{len(ref_points)}个")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                raise ValueError("标定被用户取消")
            elif key == ord('r'):  # 重置
                ref_points = []
                temp_frame = frame.copy()
                cv2.imshow('Calibration', temp_frame)
                print("标定点已重置")

        cv2.destroyAllWindows()

        # 输入实际世界坐标（单位：米）
        print(f"\n请输入{len(ref_points)}个点的实际坐标（单位：米）：")
        print("注意：X轴为横向距离（向右为正），Y轴为纵向距离（远离相机为正）")

        world_points = []
        for i, (px, py) in enumerate(ref_points):
            print(f"\n点 {i + 1} (像素坐标: {px}, {py}):")

            # 提供智能建议
            if i == 0:
                print("  建议：选择图像左下角或右下角作为原点(0,0)")
            elif i == 1:
                print("  建议：横向移动一个车道宽度（标准车道3.75米）")
            elif i >= 2:
                print("  建议：选择不同深度的点（如5米、10米、20米远处）")

            try:
                wx = float(input(f"  实际X坐标（横向距离，米）: "))
                wy = float(input(f"  实际Y坐标（纵向距离，米）: "))
            except:
                # 提供基于车道线的默认值
                if i == 0:
                    wx, wy = 0, 0
                elif i == 1:
                    wx, wy = 3.75, 0  # 车道宽度
                elif i == 2:
                    wx, wy = 0, 10  # 10米远处
                elif i == 3:
                    wx, wy = 3.75, 10
                elif i == 4:
                    wx, wy = 0, 20  # 20米远处
                elif i == 5:
                    wx, wy = 3.75, 20
                else:
                    wx, wy = (i - 5) * 2, (i - 5) * 5  # 继续增加

                print(f"  使用默认值: ({wx}, {wy})")

            world_points.append((wx, wy))

        # 计算单应性矩阵（使用RANSAC提高鲁棒性）
        pixel_points = np.float32(ref_points)
        world_points = np.float32(world_points)

        # 使用RANSAC方法，可以处理一些异常点
        self.homography_matrix, mask = cv2.findHomography(pixel_points, world_points, cv2.RANSAC, 5.0)

        # 检查内点数量
        inlier_count = np.sum(mask) if mask is not None else len(ref_points)
        print(f"内点数量: {inlier_count}/{len(ref_points)}")

        if inlier_count < min_points:
            print("警告：内点数量不足，标定可能不准确！")

        self.calibrated = True

        # 计算标定误差
        self.calibration_error = self.calculate_calibration_error(pixel_points, world_points)

        print("\n标定完成！")
        print(f"标定点数量: {len(ref_points)}")
        print(f"单应性矩阵:")
        print(self.homography_matrix)
        print(f"标定误差: {self.calibration_error:.4f} 像素")

        # 可视化标定结果
        self.visualize_calibration(frame, ref_points, world_points)

        # 保存标定结果
        self.save_calibration_data(pixel_points, world_points)

    def calculate_calibration_error(self, pixel_points, world_points):
        """计算标定误差"""
        # 将世界坐标转换回像素坐标
        world_to_pixel_matrix = np.linalg.inv(self.homography_matrix)

        errors = []
        for i, (world_point, pixel_point) in enumerate(zip(world_points, pixel_points)):
            # 世界坐标转像素坐标
            world_pt = np.array([[world_point[0], world_point[1]]], dtype=np.float32)
            predicted_pixel = cv2.perspectiveTransform(world_pt.reshape(-1, 1, 2), world_to_pixel_matrix)

            predicted_x = predicted_pixel[0][0][0]
            predicted_y = predicted_pixel[0][0][1]

            # 计算误差
            error = np.sqrt((predicted_x - pixel_point[0]) ** 2 + (predicted_y - pixel_point[1]) ** 2)
            errors.append(error)

        return np.mean(errors)

    def save_calibration_data(self, pixel_points, world_points):
        """保存标定数据到文件"""
        calib_data = {
            'pixel_points': pixel_points.tolist(),
            'world_points': world_points.tolist(),
            'homography_matrix': self.homography_matrix.tolist(),
            'calibration_error': float(self.calibration_error),
            'frame_width': self.frame_width,
            'frame_height': self.frame_height,
            'fps': self.fps
        }

        # 生成文件名
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(calib_data, f, indent=4)

        print(f"标定数据已保存到: {filename}")

    def load_calibration_data(self, filename):
        """从文件加载标定数据"""
        with open(filename, 'r') as f:
            calib_data = json.load(f)

        pixel_points = np.float32(calib_data['pixel_points'])
        world_points = np.float32(calib_data['world_points'])
        self.homography_matrix = np.array(calib_data['homography_matrix'])
        self.calibration_error = calib_data['calibration_error']
        self.calibrated = True

        print(f"从 {filename} 加载标定数据")
        print(f"标定点数量: {len(pixel_points)}")
        print(f"标定误差: {self.calibration_error:.4f} 像素")

    def visualize_calibration(self, frame, pixel_points, world_points):
        """可视化标定结果"""
        vis_frame = frame.copy()

        # 绘制标定点
        for i, (px, py) in enumerate(pixel_points):
            color = (0, 255, 0) if i < 4 else (0, 255, 255)  # 前4个点用绿色，后面的用黄色
            cv2.circle(vis_frame, (int(px), int(py)), 10, color, -1)
            cv2.putText(vis_frame, f'P{i + 1}', (int(px) + 10, int(py) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(vis_frame, f'({world_points[i][0]:.1f},{world_points[i][1]:.1f})m',
                        (int(px) + 10, int(py) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 绘制网格验证标定效果
        if len(pixel_points) >= 4:
            # 在世界坐标系中创建网格
            world_grid = []
            for x in np.arange(0, 20, 5):  # X方向：0-20米，每5米
                for y in np.arange(0, 50, 10):  # Y方向：0-50米，每10米
                    world_grid.append([x, y])

            if world_grid:
                world_grid = np.array(world_grid, dtype=np.float32)

                # 转换为像素坐标
                pixel_grid = cv2.perspectiveTransform(world_grid.reshape(-1, 1, 2),
                                                      np.linalg.inv(self.homography_matrix))

                # 绘制网格点
                for pt in pixel_grid:
                    px, py = int(pt[0][0]), int(pt[0][1])
                    if 0 <= px < self.frame_width and 0 <= py < self.frame_height:
                        cv2.circle(vis_frame, (px, py), 3, (255, 0, 0), -1)

        # 显示标定误差
        cv2.putText(vis_frame, f'Calibration Error: {self.calibration_error:.2f} pixels',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f'Points: {len(pixel_points)}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.namedWindow('Calibration Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Calibration Result', 1200, 800)
        cv2.imshow('Calibration Result', vis_frame)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    def pixel_to_world(self, pixel_point):
        """
        像素坐标转换为世界坐标

        Args:
            pixel_point: (x, y) 像素坐标

        Returns:
            (world_x, world_y) 世界坐标（米）
        """
        if not self.calibrated:
            raise ValueError("请先进行相机标定！")

        # 转换为numpy数组并确保维度正确
        pixel_point = np.array([pixel_point], dtype=np.float32)

        # 转换为世界坐标
        try:
            world_points = cv2.perspectiveTransform(pixel_point.reshape(-1, 1, 2), self.homography_matrix)
            world_x = float(world_points[0][0][0])
            world_y = float(world_points[0][0][1])

            return world_x, world_y
        except Exception as e:
            print(f"坐标转换错误: {e}")
            return 0, 0

    def validate_calibration_with_lane_width(self):
        """
        使用车道宽度验证标定准确性
        标准高速公路车道宽度为3.75米
        """
        if not self.calibrated:
            print("请先进行标定！")
            return

        print("\n=== 车道宽度验证 ===")

        # 读取一帧用于选择点
        ret, frame = self.cap.read()
        if not ret:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        print("请在图像上点击同一车道线的两个边缘点（用于验证车道宽度）")
        print("按顺序点击：左边缘 → 右边缘")

        lane_points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                lane_points.append((x, y))
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
                if len(lane_points) == 1:
                    cv2.putText(frame, "Left Edge", (x + 10, y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif len(lane_points) == 2:
                    cv2.putText(frame, "Right Edge", (x + 10, y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # 绘制连线
                    cv2.line(frame, lane_points[0], lane_points[1], (0, 255, 255), 2)
                cv2.imshow('Validate Lane Width', frame)

                if len(lane_points) == 2:
                    cv2.destroyWindow('Validate Lane Width')

        cv2.namedWindow('Validate Lane Width', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Validate Lane Width', 1200, 800)
        cv2.imshow('Validate Lane Width', frame)
        cv2.setMouseCallback('Validate Lane Width', click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(lane_points) == 2:
            # 转换为世界坐标
            world_left = self.pixel_to_world(lane_points[0])
            world_right = self.pixel_to_world(lane_points[1])

            # 计算距离
            distance = np.sqrt((world_right[0] - world_left[0]) ** 2 +
                               (world_right[1] - world_left[1]) ** 2)

            print(f"左边缘: 像素{lane_points[0]} -> 世界{world_left}")
            print(f"右边缘: 像素{lane_points[1]} -> 世界{world_right}")
            print(f"计算的车道宽度: {distance:.2f} 米")
            print(f"标准车道宽度: 3.75 米")
            print(f"误差: {abs(distance - 3.75):.2f} 米 ({abs(distance - 3.75) / 3.75 * 100:.1f}%)")

            if abs(distance - 3.75) > 0.5:  # 误差超过0.5米
                print("警告：标定误差较大，建议重新标定！")
            else:
                print("标定结果良好！")

    def init_kalman_filter(self, track_id, initial_position):
        """
        初始化卡尔曼滤波器

        Args:
            track_id: 车辆ID
            initial_position: 初始位置 (world_x, world_y)
        """
        kalman = cv2.KalmanFilter(4, 2)  # 4个状态变量，2个观测变量

        # 状态转移矩阵 (假设匀速运动)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                            [0, 1, 0, 1],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32)

        # 观测矩阵
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0]], np.float32)

        # 过程噪声协方差
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # 观测噪声协方差
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

        # 后验误差协方差
        kalman.errorCovPost = np.eye(4, dtype=np.float32)

        # 初始状态
        kalman.statePost = np.array([initial_position[0], initial_position[1], 0, 0],
                                    dtype=np.float32)

        self.kalman_filters[track_id] = kalman

    def update_kalman_filter(self, track_id, measurement):
        """
        更新卡尔曼滤波器

        Args:
            track_id: 车辆ID
            measurement: 观测位置 (world_x, world_y)

        Returns:
            filtered_position: 滤波后的位置
        """
        if track_id not in self.kalman_filters:
            self.init_kalman_filter(track_id, measurement)
            return measurement

        kalman = self.kalman_filters[track_id]

        # 预测
        prediction = kalman.predict()

        # 准备测量值
        z = np.array([[np.float32(measurement[0])],
                      [np.float32(measurement[1])]])

        # 更新
        kalman.correct(z)

        # 获取滤波后的状态
        filtered_state = kalman.statePost

        return float(filtered_state[0]), float(filtered_state[1])

    def calculate_iou(self, box1, box2):
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

    def improved_simple_tracking(self, detections, iou_threshold=0.3):
        """
        改进的追踪算法，考虑位置预测

        Args:
            detections: 当前帧的检测结果列表 [(x1, y1, x2, y2, conf, class), ...]
            iou_threshold: IoU阈值

        Returns:
            tracked_detections: 带ID的检测结果 [(x1, y1, x2, y2, conf, class, track_id), ...]
        """
        current_tracks = {}

        # 如果没有上一帧的检测，直接分配新ID
        if not self.prev_detections:
            for i, det in enumerate(detections):
                track_id = self.next_track_id
                current_tracks[track_id] = det + (track_id,)
                self.next_track_id += 1
            self.prev_detections = current_tracks.copy()
            return list(current_tracks.values())

        # 计算预测位置（简单线性预测）
        predicted_positions = {}
        for track_id, prev_det in self.prev_detections.items():
            # 如果有历史轨迹，可以进行简单预测
            if track_id in self.tracks and len(self.tracks[track_id]) >= 2:
                # 获取最后两个位置
                pos1 = self.tracks[track_id][-1]
                pos2 = self.tracks[track_id][-2] if len(self.tracks[track_id]) >= 2 else pos1

                # 计算速度
                dt = (pos1[0] - pos2[0]) / self.fps
                if dt > 0:
                    # 预测下一帧位置
                    predicted_box = list(prev_det[:4])
                    # 这里可以添加更复杂的预测逻辑
                    predicted_positions[track_id] = predicted_box

        # 计算IoU矩阵
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.prev_detections.keys())
        matched_pairs = []

        # 首先尝试匹配高IoU的
        for i in unmatched_detections[:]:
            best_iou = 0
            best_track_id = None

            for track_id in unmatched_tracks:
                # 如果有预测位置，使用预测位置计算IoU
                if track_id in predicted_positions:
                    iou = self.calculate_iou(detections[i][:4], predicted_positions[track_id])
                else:
                    iou = self.calculate_iou(detections[i][:4], self.prev_detections[track_id][:4])

                if iou > best_iou and iou > iou_threshold:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None:
                matched_pairs.append((i, best_track_id))
                unmatched_detections.remove(i)
                unmatched_tracks.remove(best_track_id)

        # 处理匹配的检测
        for det_idx, track_id in matched_pairs:
            current_tracks[track_id] = detections[det_idx] + (track_id,)

        # 为未匹配的检测分配新ID
        for det_idx in unmatched_detections:
            track_id = self.next_track_id
            current_tracks[track_id] = detections[det_idx] + (track_id,)
            self.next_track_id += 1

        # 更新历史检测
        self.prev_detections = {track_id: det[:6] for track_id, det in current_tracks.items()}

        return list(current_tracks.values())

    def improved_calculate_speed(self, track_id, current_position, frame_idx):
        """
        改进的速度计算方法，使用多帧平均和合理性检查

        Args:
            track_id: 车辆ID
            current_position: 当前世界坐标 (x, y)
            frame_idx: 当前帧索引

        Returns:
            speed_kmh: 速度（km/h）
        """
        # 获取车辆历史轨迹
        history = self.tracks.get(track_id, [])

        if len(history) < 2:
            return 0.0

        # 方法1：使用最近N个点进行线性回归（更稳定）
        if len(history) >= 2:
            # 取最近的N个点
            N = min(5, len(history))
            recent_history = history[-N:]

            # 提取时间和位置
            frames = [h[0] for h in recent_history]
            x_positions = [h[1] for h in recent_history]
            y_positions = [h[2] for h in recent_history]

            # 计算总位移和总时间
            start_idx = 0
            end_idx = len(recent_history) - 1

            # 时间差（秒）
            total_time = (frames[end_idx] - frames[start_idx]) / self.fps

            if total_time > 0:
                # 总位移
                total_distance = np.sqrt(
                    (x_positions[end_idx] - x_positions[start_idx]) ** 2 +
                    (y_positions[end_idx] - y_positions[start_idx]) ** 2
                )

                # 计算平均速度
                speed_ms = total_distance / total_time
                speed_kmh = speed_ms * 3.6

                # 方法2：使用两帧计算作为备选
                last_frame_idx, last_x, last_y, _ = history[-1]
                prev_frame_idx, prev_x, prev_y, _ = history[-2]

                dt2 = (last_frame_idx - prev_frame_idx) / self.fps
                distance2 = np.sqrt((last_x - prev_x) ** 2 + (last_y - prev_y) ** 2)
                speed_kmh2 = (distance2 / dt2) * 3.6 if dt2 > 0 else 0

                # 两种方法的加权平均
                final_speed = speed_kmh * 0.7 + speed_kmh2 * 0.3
        else:
            # 使用最近的两个位置计算速度
            last_frame_idx, last_x, last_y, _ = history[-1]
            prev_frame_idx, prev_x, prev_y, _ = history[-2]

            # 计算时间差（秒）
            dt = (last_frame_idx - prev_frame_idx) / self.fps

            if dt <= 0:
                return 0.0

            # 计算距离（米）
            distance = np.sqrt((last_x - prev_x) ** 2 + (last_y - prev_y) ** 2)

            # 计算速度（m/s）
            speed_ms = distance / dt

            # 转换为km/h
            final_speed = speed_ms * 3.6

        # 合理性检查
        if not self.is_speed_reasonable(final_speed, track_id):
            # 如果不合理，返回平滑后的值
            if len(self.speeds[track_id]) > 0:
                final_speed = np.mean(list(self.speeds[track_id])[-5:]) if len(
                    self.speeds[track_id]) >= 5 else final_speed

        # 添加到速度历史用于平滑
        self.speeds[track_id].append(final_speed)

        # 使用滑动平均平滑速度
        if len(self.speeds[track_id]) > 0:
            smoothed_speed = np.mean(self.speeds[track_id])
        else:
            smoothed_speed = final_speed

        return smoothed_speed

    def is_speed_reasonable(self, speed_kmh, track_id):
        """
        检查速度是否合理
        """
        # 高速公路速度限制
        MIN_SPEED = 0  # 最低20km/h（考虑到匝道车辆）
        MAX_SPEED = 150  # 最大150km/h（考虑到可能的误差）

        if speed_kmh < MIN_SPEED or speed_kmh > MAX_SPEED:
            return False

        # 检查加速度是否合理（最大10m/s²）
        if track_id in self.speeds and len(self.speeds[track_id]) >= 2:
            recent_speeds = list(self.speeds[track_id])[-2:]
            if len(recent_speeds) >= 2:
                acceleration = abs(recent_speeds[-1] - recent_speeds[-2]) / (1 / self.fps)  # km/h per second
                if acceleration > 50:  # 约13.9 m/s²
                    return False

        return True

    def process_frame(self, frame, frame_idx):
        """
        处理单帧图像

        Args:
            frame: 输入帧
            frame_idx: 帧索引

        Returns:
            processed_frame: 处理后的帧
            speeds_info: 速度信息字典
        """
        if frame is None:
            return None, {}

        # 使用YOLOv11进行检测
        results = self.model(frame, verbose=False, save=False, save_txt=False, save_conf=False)[0]

        # 过滤出车辆检测
        vehicle_detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.vehicle_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    vehicle_detections.append((x1, y1, x2, y2, conf, cls_id))

        # 使用改进的追踪算法
        tracked_detections = self.improved_simple_tracking(vehicle_detections)

        speeds_info = {}
        processed_frame = frame.copy()

        # 处理每个追踪到的车辆
        for det in tracked_detections:
            x1, y1, x2, y2, conf, cls_id, track_id = det

            # 使用底部中心点（关键！）
            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2  # 使用底部的y坐标

            # 转换为世界坐标
            if self.calibrated:
                try:
                    world_x, world_y = self.pixel_to_world((bottom_center_x, bottom_center_y))

                    # 使用卡尔曼滤波平滑位置
                    filtered_x, filtered_y = self.update_kalman_filter(track_id, (world_x, world_y))

                    # 计算速度（使用改进的方法）
                    speed_kmh = self.improved_calculate_speed(track_id, (filtered_x, filtered_y), frame_idx)

                    # 保存轨迹
                    self.tracks[track_id].append((frame_idx, filtered_x, filtered_y, speed_kmh))

                    # 限制轨迹长度
                    if len(self.tracks[track_id]) > 100:
                        self.tracks[track_id].pop(0)

                    # 记录速度信息
                    speeds_info[track_id] = {
                        'speed': speed_kmh,
                        'position': (filtered_x, filtered_y),
                        'pixel_position': (bottom_center_x, bottom_center_y),
                        'class_id': cls_id
                    }

                    # 可视化
                    self.visualize_vehicle(processed_frame, det, speed_kmh, track_id)

                except Exception as e:
                    print(f"处理车辆{track_id}时出错: {e}")
                    continue

        return processed_frame, speeds_info

    def visualize_vehicle(self, frame, detection, speed, track_id):
        """
        在帧上可视化车辆和速度信息
        """
        x1, y1, x2, y2, conf, cls_id, track_id = detection

        # 使用随机颜色或速度相关颜色
        if hasattr(self, 'colors'):
            color = self.colors[track_id % len(self.colors)]

        # 绘制边界框
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 绘制ID和速度
        label = f"ID:{track_id} {speed:.0f}km/h"

        # 获取文本大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        # 绘制文本背景
        text_bg_top = int(y1) - text_size[1] - 10
        text_bg_bottom = int(y1)
        text_bg_left = int(x1)
        text_bg_right = int(x1) + text_size[0] + 10

        # 确保文本框在图像范围内
        if text_bg_top < 0:
            text_bg_top = int(y2) + 10
            text_bg_bottom = text_bg_top + text_size[1] + 10

        cv2.rectangle(frame,
                      (text_bg_left, text_bg_top),
                      (text_bg_right, text_bg_bottom),
                      color, -1)

        # 绘制文本
        cv2.putText(frame, label,
                    (text_bg_left + 5, text_bg_bottom - 5),
                    font, font_scale, (255, 255, 255), thickness)

        # 绘制底部中心点
        bottom_center_x = int((x1 + x2) / 2)
        bottom_center_y = int(y2)
        cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, (0, 0, 255), -1)

    def run(self, output_path=None, display=True, max_frames=None, validate_calibration=False):
        """
        运行主处理流程

        Args:
            output_path: 输出视频路径（如果为None则不保存）
            display: 是否显示处理结果
            max_frames: 最大处理帧数
            validate_calibration: 是否验证标定
        """
        # 重置视频到开始
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 如果没有标定，进行标定
        if not self.calibrated:
            ret, frame = self.cap.read()
            if ret:
                print("\n开始标定过程...")
                self.calibrate_camera(frame)

                # 验证标定
                if validate_calibration:
                    self.validate_calibration_with_lane_width()

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 准备视频写入器
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps,
                                  (self.frame_width, self.frame_height))

        frame_idx = 0
        all_speeds = []
        speed_history = defaultdict(list)

        print("\n开始处理视频...")

        while True:
            if max_frames and frame_idx >= max_frames:
                break

            ret, frame = self.cap.read()
            if not ret:
                break

            # 处理当前帧
            processed_frame, speeds_info = self.process_frame(frame, frame_idx)

            # 收集速度信息
            for track_id, info in speeds_info.items():
                speed = info['speed']
                all_speeds.append(speed)
                speed_history[track_id].append(speed)

            # 添加帧信息
            info_text = f"Frame: {frame_idx} | Vehicles: {len(speeds_info)}"
            cv2.putText(processed_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 添加速度统计
            if all_speeds:
                # 使用最近50个速度计算平均值
                recent_speeds = all_speeds[-50:] if len(all_speeds) > 50 else all_speeds
                avg_speed = np.mean(recent_speeds)

                cv2.putText(processed_frame, f"Avg Speed: {avg_speed:.1f} km/h", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # 显示标定误差
                if self.calibration_error is not None:
                    cv2.putText(processed_frame, f"Cal Error: {self.calibration_error:.2f}px",
                                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 显示处理结果
            if display:
                cv2.namedWindow('Vehicle Speed Detection', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Vehicle Speed Detection', 1200, 800)
                cv2.imshow('Vehicle Speed Detection', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 保存输出视频
            if output_path:
                out.write(processed_frame)

            frame_idx += 1

            # 显示进度
            if frame_idx % 10 == 0:
                print(f"已处理 {frame_idx} 帧, 检测到 {len(speeds_info)} 辆车")

        # 清理
        self.cap.release()
        if output_path:
            out.release()
        if display:
            cv2.destroyAllWindows()

        # 打印统计信息
        print("\n=== 处理完成 ===")
        print(f"总帧数: {frame_idx}")
        print(f"检测到的车辆数: {len(self.tracks)}")
        if all_speeds:
            print(f"平均速度: {np.mean(all_speeds):.1f} km/h")
            print(f"最大速度: {np.max(all_speeds):.1f} km/h")
            print(f"最小速度: {np.min(all_speeds):.1f} km/h")

        return self.tracks, all_speeds

# 改进的主函数
def main():

    # 获取视频路径
    video_path = input("请输入视频文件路径: ").strip()
    if not video_path:
        video_path = "Dataset/Highway.mp4"

    try:
        # 检查是否有保存的标定文件
        load_calib = input("是否加载已有的标定文件？(y/n): ").strip().lower()

        if load_calib == 'y':
            calib_file = input("请输入标定文件路径: ").strip()
            if os.path.exists(calib_file):
                estimator = VehicleSpeedEstimator(video_path, fps=30)
                estimator.load_calibration_data(calib_file)
            else:
                print(f"文件不存在: {calib_file}")
                print("将进行新标定...")
                estimator = VehicleSpeedEstimator(video_path, fps=30)
        else:
            estimator = VehicleSpeedEstimator(video_path, fps=30)

        max_frames = None

        # 是否验证标定
        validate = input("是否验证标定？(y/n): ").strip().lower() == 'y'

        # 运行处理
        tracks, speeds = estimator.run(
            output_path="Output/output.mp4",
            display=True,
            max_frames=max_frames,
            validate_calibration=validate
        )

        # 保存速度数据
        if tracks:
            print("\n车辆速度数据（前10辆车）:")
            vehicle_count = 0
            for track_id, history in tracks.items():
                if history and vehicle_count < 10:
                    last_speed = history[-1][3]
                    avg_speed = np.mean([h[3] for h in history if h[3] > 0])
                    print(f"  车辆 {track_id}: 最后速度={last_speed:.1f} km/h, 平均={avg_speed:.1f} km/h")
                    vehicle_count += 1

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
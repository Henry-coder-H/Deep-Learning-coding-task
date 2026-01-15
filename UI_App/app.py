"""
智能交通识别系统 - Streamlit UI
集成车型识别、车牌识别、车速识别三大核心功能
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import colorsys
from streamlit_image_coordinates import streamlit_image_coordinates
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置页面配置
st.set_page_config(
    page_title="智能交通识别系统",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 注入 CSS 样式来“汉化”上传组件 ---
# --- 注入 CSS 样式来“汉化”上传组件 (v2.0 优化版) ---
st.markdown("""
<style>
    /* 1. 隐藏原来的 "Drag and drop file here" 和 "Limit 200MB..." */
    [data-testid='stFileUploaderDropzone'] div div span,
    [data-testid='stFileUploaderDropzone'] div div small {
       display: none;
    }
    
    /* 2. 自定义中间提示文本 */
    [data-testid='stFileUploaderDropzone'] div div::after {
       content: "点击上传文件或将文件拖拽至此处";
       font-size: 16px; /* 💡 修改这里：调小了字体 (原来是 1.2em) */
       margin-bottom: 10px;
    }

    /* 3. 强行修改 "Browse files" 按钮文字 */
    /* 第一步：把按钮里的原英文变透明/隐藏 */
    [data-testid='stFileUploaderDropzone'] button {
        font-size: 0 !important; /* 将原字体设为0，相当于隐藏 */
        min-width: 80px; /* 保证按钮宽度 */
    }
    
    /* 第二步：在按钮上用伪元素“写”上中文 */
    [data-testid='stFileUploaderDropzone'] button::after {
        content: "上传文件"; /* ✨ 这里改按钮文字 */
        font-size: 14px !important; /* 恢复字体大小 */
        color: inherit; /* 跟随主题颜色 */
        visibility: visible;
        display: block;
        padding-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# 配置路径
WEIGHTS_DIR = Path(__file__).parent / "weights"
TEMP_DIR = Path(tempfile.gettempdir()) / "traffic_recognition"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# 支持的文件格式
IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png']
VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov']


def get_distinct_colors(n: int) -> List[Tuple[int, int, int]]:
    """生成n个视觉上可区分的颜色"""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.8 + (i % 2) * 0.1
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))  # BGR
    return colors

# 预生成100种颜色用于车辆ID
VEHICLE_COLORS = get_distinct_colors(100)


def get_vehicle_color(track_id: int) -> Tuple[int, int, int]:
    """根据车辆ID获取固定颜色"""
    return VEHICLE_COLORS[track_id % len(VEHICLE_COLORS)]


def get_chinese_font(size: int = 20):
    """获取支持中文的字体"""
    # 常见的中文字体路径
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
        "/System/Library/Fonts/PingFang.ttc",  # macOS
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    
    # 如果找不到中文字体，返回默认字体
    return ImageFont.load_default()


def put_chinese_text(img: np.ndarray, text: str, position: Tuple[int, int], 
                     font_size: int = 20, color: Tuple[int, int, int] = (255, 255, 255),
                     bg_color: Tuple[int, int, int] = None) -> np.ndarray:
    """
    在图像上绘制支持中文的文本
    
    Args:
        img: BGR图像
        text: 要绘制的文本
        position: (x, y) 左上角位置
        font_size: 字体大小
        color: 文字颜色 (BGR)
        bg_color: 背景颜色 (BGR), None表示无背景
    
    Returns:
        绘制后的图像
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_chinese_font(font_size)
    
    # PIL使用RGB颜色
    text_color = (color[2], color[1], color[0])
    
    # 绘制背景
    if bg_color is not None:
        bg_rgb = (bg_color[2], bg_color[1], bg_color[0])
        bbox = draw.textbbox(position, text, font=font)
        padding = 3
        draw.rectangle([bbox[0] - padding, bbox[1] - padding, 
                       bbox[2] + padding, bbox[3] + padding], fill=bg_rgb)
    
    draw.text(position, text, font=font, fill=text_color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# app.py

# 1. 原有的 Torch 模型加载器 (给图片用)
@st.cache_resource
def load_torch_plate_recognizer():
    """加载原版 PyTorch 车牌识别模型 (用于图片)"""
    from models.plate_recognizer import PlateRecognizer  # 你的旧模型文件
    
    det_weights = WEIGHTS_DIR / "plate_detect.pt"
    rec_weights = WEIGHTS_DIR / "plate_rec.pth"
    
    # 检查权重是否存在
    if not det_weights.exists() or not rec_weights.exists():
        # 如果是图片模式报错，我们只在日志里提示，不阻断视频功能
        print(f"Torch权重缺失: {det_weights} 或 {rec_weights}")
        return None
        
    try:
        device = "cuda:0" if is_cuda_available() else "cpu"
        return PlateRecognizer(str(det_weights), str(rec_weights), device=device)
    except Exception as e:
        st.error(f"加载 Torch 车牌模型失败: {e}")
        return None


@st.cache_resource
def load_paddle_plate_recognizer():
    """加载 Paddle 视频专用车牌识别模型 (基于 test.py)"""
    from models.paddle_model import PaddleVideoRecognizer
    
    # 确保 yolov8n.pt 在 weights 目录下
    yolo_weights = WEIGHTS_DIR / "yolov8n.pt"
    
    if not yolo_weights.exists():
        st.error(f"❌ 视频识别需要 yolov8n.pt，请检查 {WEIGHTS_DIR}")
        return None
        
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        return PaddleVideoRecognizer(str(yolo_weights), use_gpu=use_gpu)
    except Exception as e:
        st.error(f"加载 Paddle 视频模型失败: {e}")
        return None

@st.cache_resource
def load_speed_estimator():
    """加载车速估计器 (缓存)"""
    from models.speed_estimator import VehicleSpeedEstimator
    
    vehicle_weights = WEIGHTS_DIR / "yolov11l.pt"
    
    if not vehicle_weights.exists():
        st.error(f"❌ 车辆检测模型权重不存在！请将 yolov11l.pt 放置在 {WEIGHTS_DIR} 目录下")
        return None
        
    try:
        return VehicleSpeedEstimator(fps=30.0, vehicle_model_path=str(vehicle_weights))
    except Exception as e:
        st.error(f"加载车速估计器失败: {e}")
        return None


@st.cache_resource
def load_vehicle_classifier():
    """加载车型分类器 (缓存)"""
    from models.vehicle_classifier import VehicleTypeClassifier
    
    # 指向你的 Task1 训练好的权重
    # 请确保将 task1/best.pt 复制到 UI_App/weights/best.pt
    classifier_weights = WEIGHTS_DIR / "best.pt" 
    
    if classifier_weights.exists():
        return VehicleTypeClassifier(str(classifier_weights))
    else:
        st.warning(f"⚠️ 未找到车型识别权重: {classifier_weights}，请上传文件。")
        return VehicleTypeClassifier() # 空模型，防止报错


def is_cuda_available() -> bool:
    """检查CUDA是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def is_image_file(filename: str) -> bool:
    """判断是否为图片文件"""
    ext = filename.lower().split('.')[-1]
    return ext in IMAGE_EXTENSIONS


def is_video_file(filename: str) -> bool:
    """判断是否为视频文件"""
    ext = filename.lower().split('.')[-1]
    return ext in VIDEO_EXTENSIONS

def associate_plates_to_vehicles(vehicles: List[Dict], plates: List[Dict]) -> List[Dict]:
    """
    将车牌检测结果分配给车辆检测结果 (基于中心点包含关系)
    
    Args:
        vehicles: 车辆检测结果列表 (需包含 bbox 字段)
        plates: 车牌检测结果列表 (需包含 bbox, text, conf 字段)
    
    Returns:
        合并后的检测结果列表
    """
    if not vehicles:
        # 如果没有检测到车，把所有车牌作为独立对象返回
        return [{'bbox': p['bbox'], 'plate_text': p['text'], 'plate_conf': p['conf'], 'track_id': -1} for p in plates]
    
    # 1. 深拷贝车辆列表，作为最终结果的基础
    # 注意：我们要保留原始的 vehicle 字典结构
    merged_results = [v.copy() for v in vehicles]
    
    # 2. 遍历所有车牌，尝试匹配车辆
    for plate in plates:
        px1, py1, px2, py2 = plate.get('bbox')
        p_cx = (px1 + px2) / 2
        p_cy = (py1 + py2) / 2
        
        matched = False
        for vehicle in merged_results:
            vx1, vy1, vx2, vy2 = vehicle['bbox']
            
            # 核心逻辑：判定车牌中心是否在车辆框内
            if vx1 < p_cx < vx2 and vy1 < p_cy < vy2:
                # 匹配成功！将车牌信息注入到该车辆字典中
                vehicle['plate_text'] = plate.get('text', '')
                vehicle['plate_conf'] = plate.get('conf', 0)
                # 标记已匹配
                matched = True
                break # 一个车牌只能归属一辆车，找到后跳出
        
        # 3. 处理未匹配的孤立车牌 (例如车没识别出来，但识别到了牌)
        if not matched:
            merged_results.append({
                'bbox': plate.get('bbox'),
                'plate_text': plate.get('text', ''),
                'plate_conf': plate.get('conf', 0),
                'track_id': -1, # 孤立车牌没有车辆ID
                'vehicle_type': 'Unknown', # 可选
                'conf': 0.0
            })
            
    return merged_results

def draw_detection_results(image: np.ndarray, detections: List[Dict], 
                           show_plate: bool = True, show_type: bool = True,
                           show_speed: bool = False) -> np.ndarray:
    """
    在图像上绘制检测结果（支持中文显示）
    
    Args:
        image: 输入图像 (BGR)
        detections: 检测结果列表
        show_plate: 是否显示车牌
        show_type: 是否显示车型
        show_speed: 是否显示车速
    """
    vis = image.copy()
    
    for det in detections:
        track_id = det.get('track_id', 0)
        bbox = det.get('bbox', None)
        
        if bbox is None:
            continue
            
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # 根据车辆ID分配颜色
        color = get_vehicle_color(track_id)
        
        # 绘制边界框
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        
        # 构建标签文本
        labels = []
        if show_type and 'vehicle_type' in det:
            labels.append(det['vehicle_type'])
        if show_plate and 'plate_text' in det:
            labels.append(det['plate_text'])
        if show_speed and 'speed' in det:
            labels.append(f"{det['speed']:.0f}km/h")
            
        if labels:
            label_text = " | ".join(labels)
            
            # 计算文本位置
            text_y = y1 - 5 if y1 > 30 else y2 + 25
            
            # 使用支持中文的绘制函数
            vis = put_chinese_text(vis, label_text, (x1, text_y - 20), 
                                   font_size=20, color=(255, 255, 255), bg_color=color)
            
    return vis

def draw_statistics_charts(vehicle_counts, time_series_data=None):
    """
    绘制统计图表
    Args:
        vehicle_counts: dict, {车型: 数量}
        time_series_data: list, [{'time': t, 'count': c}, ...] (仅视频需要)
    """
    if not vehicle_counts:
        st.info("暂无统计数据")
        return

    # 准备数据
    labels = list(vehicle_counts.keys())
    sizes = list(vehicle_counts.values())
    
    # 颜色映射 (与 OpenCV 绘图保持一致，转为 Hex 或 RGB 0-1)
    # 这里为了简单，使用 matplotlib 默认或自定义一组
    
    if time_series_data is not None:
        # === 视频模式：双图 (饼图 + 折线图) ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 饼图
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
        ax1.set_title("车型分布比例")
        
        # 2. 折线图
        if time_series_data:
            df_time = pd.DataFrame(time_series_data)
            ax2.plot(df_time['time'], df_time['count'], marker='o', linestyle='-', color='b', linewidth=2)
            ax2.fill_between(df_time['time'], df_time['count'], color='skyblue', alpha=0.3)
            ax2.set_xlabel("时间 (s)")
            ax2.set_ylabel("累计车辆总数")
            ax2.set_title("车流量随时间趋势")
            ax2.grid(True, linestyle='--', alpha=0.7)
            
    else:
        # === 图片模式：单图 (饼图) ===
        fig, ax1 = plt.subplots(figsize=(6, 6))
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, shadow=True)
        ax1.set_title("车型识别分布比例")
    
    st.pyplot(fig)
    plt.close(fig) # 释放内存

def process_image(image: np.ndarray, enable_plate: bool, enable_type: bool) -> Tuple[np.ndarray, pd.DataFrame]:
    """处理单张图片 (已添加合并逻辑)"""
    vehicle_detections = []
    plate_raw_results = [] # 暂存原始车牌结果
    
    vehicle_counts = defaultdict(int) 
    
    # --- 1. 车型识别 (收集车辆框) ---
    vehicle_classifier = load_vehicle_classifier() if enable_type else None
    if enable_type and vehicle_classifier:
        type_results = vehicle_classifier.predict(image)
        for res in type_results:
            # 构造标准车辆对象
            vehicle_detections.append({
                'bbox': res['bbox'],
                'vehicle_type': res['class_name'],
                'track_id': -1, 
                'conf': res['conf']
            })
            vehicle_counts[res['class_name']] += 1
            
    # --- 2. 车牌识别 (收集车牌结果) ---
    if enable_plate:
        plate_recognizer = load_torch_plate_recognizer()
        if plate_recognizer:
            plate_raw_results = plate_recognizer.recognize_image(image)
    
    # --- 3. 执行合并逻辑 ---
    # 如果开启了车型识别，尝试将车牌归并到车辆中；否则直接显示车牌
    if enable_type and vehicle_detections:
        final_detections = associate_plates_to_vehicles(vehicle_detections, plate_raw_results)
    else:
        # 如果没开车型识别，或者没检测到车，直接转换车牌格式
        final_detections = vehicle_detections # 先包含已有的(可能是空的)
        for p in plate_raw_results:
            final_detections.append({
                'bbox': p['bbox'],
                'plate_text': p['text'],
                'plate_conf': p['conf'],
                'track_id': -1
            })

    # --- 4. 绘制结果 ---
    # draw_detection_results 会自动处理字典里同时有 vehicle_type 和 plate_text 的情况
    vis_image = draw_detection_results(image, final_detections, show_plate=enable_plate, show_type=enable_type)
    
    # --- 5. 生成统计表格 (现在一行数据会同时包含车型和车牌) ---
    table_data = []
    for det in final_detections:
        row = {}
        # 只有当检测结果包含相关信息时才加入表格
        has_info = False
        
        if 'vehicle_type' in det:
             row['类型'] = det['vehicle_type']
             row['类型置信度'] = f"{det['conf']:.2f}"
             has_info = True
             
        if 'plate_text' in det:
             row['车牌'] = det['plate_text']
             row['车牌置信度'] = f"{det['plate_conf']:.2f}" # 可选
             has_info = True
        
        if has_info: 
            table_data.append(row)
        
    df = pd.DataFrame(table_data)
    
    st.session_state.temp_vehicle_counts = vehicle_counts
    st.session_state.temp_time_series = None
    
    return vis_image, df


def process_video(video_path: str, enable_plate: bool, enable_type: bool, enable_speed: bool,
                  speed_estimator=None, progress_callback=None) -> str:
    """处理视频 (已添加合并逻辑)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"无法打开视频: {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = str(TEMP_DIR / f"output_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    plate_recognizer = load_paddle_plate_recognizer() if enable_plate else None
    vehicle_classifier = load_vehicle_classifier() if enable_type else None
    
    if enable_speed and speed_estimator:
        speed_estimator.fps = fps
        speed_estimator.set_frame_size(width, height)
        speed_estimator.reset()
        
    unique_vehicle_ids = defaultdict(set) 
    time_series_data = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        current_vehicles = []
        current_plates = []
        
        # --- A. 车型识别 (获取带 ID 的车辆) ---
        if enable_type and vehicle_classifier:
            type_results = vehicle_classifier.track(frame)
            for res in type_results:
                tid = res['track_id']
                cls_name = res['class_name']
                if tid != -1:
                    unique_vehicle_ids[cls_name].add(tid)
                
                current_vehicles.append({
                    'track_id': tid,
                    'bbox': res['bbox'],
                    'vehicle_type': cls_name,
                    'conf': res['conf']
                })

        # --- B. 车牌识别 ---
        if enable_plate and plate_recognizer:
            # 这里的 recognize_image 返回的是 list[dict]
            current_plates = plate_recognizer.recognize_image(frame)

        # --- C. 合并逻辑 ---
        if enable_type and current_vehicles:
            final_detections = associate_plates_to_vehicles(current_vehicles, current_plates)
        else:
            # 没有车或者没开车型识别，只显示车牌
            final_detections = current_vehicles # 包含空的或者仅有车的(如果有逻辑漏洞的话)
            for p in current_plates:
                final_detections.append({
                    'bbox': p['bbox'],
                    'plate_text': p['text'],
                    'track_id': -1
                })
        
        # --- D. 车速识别 (单独处理，追加到列表) ---
        if enable_speed and speed_estimator and speed_estimator.calibrated:
            _, speeds_info = speed_estimator.process_frame(frame, frame_idx)
            for track_id, info in speeds_info.items():
                # 注意：这里可能会产生重叠框，因为车速模块有自己的检测器
                # 完美方案是将车速模块的ID与Task1的ID对齐，但这比较复杂。
                # 现在的处理是作为额外的框绘制。
                final_detections.append({
                    'track_id': track_id,
                    'bbox': info['bbox'],
                    'speed': info['speed']
                })

        # --- E. 统计与绘制 ---
        if frame_idx % int(fps) == 0:
            current_total = sum(len(ids) for ids in unique_vehicle_ids.values())
            time_series_data.append({'time': frame_idx / fps, 'count': current_total})

        vis_frame = draw_detection_results(frame, final_detections, 
                                           show_plate=enable_plate, 
                                           show_type=enable_type,
                                           show_speed=enable_speed)
        
        if enable_type:
            y_offset = 30
            for cls_name in sorted(unique_vehicle_ids.keys()):
                count = len(unique_vehicle_ids[cls_name])
                color = vehicle_classifier.get_color(cls_name)
                cv2.putText(vis_frame, f"{cls_name}: {count}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 30

        writer.write(vis_frame)
        frame_idx += 1
        if progress_callback: progress_callback(frame_idx / total_frames)
            
    cap.release()
    writer.release()
    
    final_counts = {k: len(v) for k, v in unique_vehicle_ids.items()}
    st.session_state.temp_vehicle_counts = final_counts
    st.session_state.temp_time_series = time_series_data
    
    return output_path


def show_calibration_guide():
    """显示标定指南"""
    with st.expander("📖 坐标标定参考指南", expanded=True):
        st.markdown("""
        ### 操作说明
        
        1. 在视频画面中点击选择关键特征点（如车道线端点、斑马线角点），将其标记为坐标原点 $(0, 0)$
        2. 根据下方的参考数据，再选取别的参考点，估算该点相对于原点的距离(m)标注 $(X, Y)$
        3. 请您尽可能多的标注特征点（建议至少6个点）
        
        ### 参考数据
        
        #### 车行道分界线
        - **高速公路**：6-9线（线段长度6m，间隔9m）
        - **城市快速路**：4-4线或4-6线（线段长度4m，间隔4m或6m）
        
        #### 车道宽度参考表
        
        | 道路类型 | 标准宽度 (m) | 最小值 (m) |
        |---------|:-----------:|:---------:|
        | 高速公路 | **3.75** | 3.50 |
        | 一级/二级公路 | **3.75** | 3.50 |
        | 城市快速路 | **3.75** | 3.50 |
        | 城市次干路 | **3.50** | 3.25 |
        | 城市支路 | **3.25** | 2.80 |
        
        #### 人行横道线
        - 最小宽度：3m
        - 可按行人流量以1m为单位加宽
        """)


def draw_calibration_points_on_image(image: np.ndarray, points: List[dict], pending_point: tuple = None) -> np.ndarray:
    """
    在图像上绘制已标定的点
    
    Args:
        image: 输入图像 (BGR)
        points: 标定点列表
        pending_point: 待确认的点 (px, py)，用黄色显示
    
    Returns:
        绘制后的图像
    """
    vis = image.copy()
    
    # 绘制已确认的点（绿色）
    for i, pt in enumerate(points):
        px, py = int(pt['px']), int(pt['py'])
        # 绘制圆点
        cv2.circle(vis, (px, py), 8, (0, 255, 0), -1)
        cv2.circle(vis, (px, py), 10, (255, 255, 255), 2)
        # 绘制标号
        label = f"P{i+1}"
        cv2.putText(vis, label, (px + 12, py + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, label, (px + 12, py + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # 绘制待确认的点（黄色）
    if pending_point is not None:
        px, py = int(pending_point[0]), int(pending_point[1])
        cv2.circle(vis, (px, py), 10, (0, 255, 255), -1)
        cv2.circle(vis, (px, py), 12, (255, 255, 255), 2)
        label = f"P{len(points)+1}?"
        cv2.putText(vis, label, (px + 14, py + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, label, (px + 14, py + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    return vis


def calibration_interface(first_frame: np.ndarray, speed_estimator) -> bool:
    """
    标定界面 - 支持交互式点击选点
    
    Returns:
        bool: 标定是否完成
    """
    st.subheader("🎯 距离标定")
    
    # 显示指南
    show_calibration_guide()
    
    # 初始化 session_state
    if 'calib_points' not in st.session_state:
        st.session_state.calib_points = []  # 已确认的标定点列表
    if 'pending_click' not in st.session_state:
        st.session_state.pending_click = None  # 待确认的点击坐标 (px, py)
    if 'last_click_coords' not in st.session_state:
        st.session_state.last_click_coords = None
    
    # 获取原始图像尺寸
    orig_height, orig_width = first_frame.shape[:2]
    
    # ===== 交互式点击区域 =====
    st.markdown("### 📍 点击图片添加标定点")
    st.info("💡 **操作步骤**：点击图片选择一个特征点 → 输入该点的相对坐标(米) → 点击「确认添加」")
    
    # 在图像上绘制已标定的点和待确认的点
    display_image = draw_calibration_points_on_image(
        first_frame, 
        st.session_state.calib_points,
        st.session_state.pending_click
    )
    
    # 转换为 RGB 用于显示
    display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(display_image_rgb)
    
    # 计算显示尺寸（保持宽高比，最大宽度800）
    max_display_width = 800
    scale = min(max_display_width / orig_width, 1.0)
    display_width = int(orig_width * scale)
    display_height = int(orig_height * scale)
    
    # 使用 streamlit_image_coordinates 获取点击坐标
    clicked_coords = streamlit_image_coordinates(
        pil_image,
        width=display_width,
        height=display_height,
        key="calibration_image"
    )
    
    # 处理点击事件
    if clicked_coords is not None:
        click_x = int(clicked_coords['x'] / scale)
        click_y = int(clicked_coords['y'] / scale)
        current_click = (click_x, click_y)
        
        # 只有当这是一个新的点击时才更新待确认点
        if st.session_state.last_click_coords != current_click:
            st.session_state.last_click_coords = current_click
            st.session_state.pending_click = current_click
            st.rerun()
    
    st.caption(f"图像尺寸: {orig_width} x {orig_height} 像素 | 已标定 {len(st.session_state.calib_points)} 个点")
    
    # ===== 添加新点的输入区域 =====
    if st.session_state.pending_click is not None:
        st.markdown("---")
        st.markdown(f"### ➕ 添加新标定点 P{len(st.session_state.calib_points) + 1}")
        
        px, py = st.session_state.pending_click
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📍 像素坐标（已从图片获取）**")
            disp_col1, disp_col2 = st.columns(2)
            disp_col1.metric("像素 X", px)
            disp_col2.metric("像素 Y", py)
        
        with col2:
            st.markdown("**🌍 相对坐标（请输入，单位：米）**")
            input_col1, input_col2 = st.columns(2)
            world_x = input_col1.number_input("X (m)", value=0.0, format="%.1f", key="new_wx")
            world_y = input_col2.number_input("Y (m)", value=0.0, format="%.1f", key="new_wy")
        
        # 添加和取消按钮
        btn_col1, btn_col2 = st.columns(2)
        
        if btn_col1.button("✅ 确认添加", type="primary", key="add_point_btn"):
            # 添加新点到列表
            new_point = {
                'px': px,
                'py': py,
                'wx': world_x,
                'wy': world_y
            }
            st.session_state.calib_points.append(new_point)
            st.session_state.pending_click = None
            st.toast(f"✅ P{len(st.session_state.calib_points)} 已添加")
            st.rerun()
        
        if btn_col2.button("❌ 取消", key="cancel_point_btn"):
            st.session_state.pending_click = None
            st.rerun()
    else:
        st.info("👆 请点击上方图片选择一个标定点")
    
    # ===== 已确认的标定点列表 =====
    st.markdown("---")
    st.markdown("### 📋 已添加的标定点")
    
    if len(st.session_state.calib_points) == 0:
        st.warning("暂无标定点，请点击图片添加")
    else:
        # 表头
        cols_header = st.columns([1, 2, 2, 2, 2, 1])
        cols_header[0].write("**点**")
        cols_header[1].write("**像素X**")
        cols_header[2].write("**像素Y**")
        cols_header[3].write("**相对X(m)**")
        cols_header[4].write("**相对Y(m)**")
        cols_header[5].write("**操作**")
        
        # 显示每个标定点
        points_to_remove = []
        for i, pt in enumerate(st.session_state.calib_points):
            cols = st.columns([1, 2, 2, 2, 2, 1])
            cols[0].write(f"**P{i+1}**")
            cols[1].write(f"{int(pt['px'])}")
            cols[2].write(f"{int(pt['py'])}")
            cols[3].write(f"{pt['wx']:.1f}")
            cols[4].write(f"{pt['wy']:.1f}")
            
            if cols[5].button("🗑️", key=f"del_btn_{i}", help="删除此点"):
                points_to_remove.append(i)
        
        # 删除标记的点
        if points_to_remove:
            for idx in sorted(points_to_remove, reverse=True):
                st.session_state.calib_points.pop(idx)
            st.rerun()
    
    # ===== 最终操作按钮 =====
    st.markdown("---")
    
    col_btn1, col_btn2 = st.columns(2)
    
    # 清除所有点按钮
    if col_btn1.button("🗑️ 清除所有标定点", key="clear_all_points"):
        st.session_state.calib_points = []
        st.session_state.pending_click = None
        st.session_state.last_click_coords = None
        st.session_state.calibration_step = 'adding_points'
        st.rerun()
    
    # 确认标定按钮
    num_valid_points = len(st.session_state.calib_points)
    can_calibrate = num_valid_points >= 4
    
    if col_btn2.button("✅ 完成标定", type="primary", key="confirm_calib", disabled=not can_calibrate):
        # 收集标定点
        pixel_points = [(pt['px'], pt['py']) for pt in st.session_state.calib_points]
        world_points = [(pt['wx'], pt['wy']) for pt in st.session_state.calib_points]
        
        # 执行标定
        success = speed_estimator.calibrate_from_points(pixel_points, world_points)
        
        if success:
            st.session_state.calibration_step = 'ask_validation'
            st.session_state.temp_calibration_error = speed_estimator.calibration_error
            st.session_state.temp_num_points = num_valid_points
            st.rerun()
        else:
            st.error("标定失败，请检查输入的点是否正确")
            return False
    
    if not can_calibrate:
        st.warning(f"⚠️ 需要至少 4 个标定点才能完成标定（当前: {num_valid_points} 个）")
            
    return False


def validation_interface(first_frame: np.ndarray, speed_estimator) -> bool:
    """
    验证标定界面 - 在标定完成后显示
    
    Returns:
        bool: 是否完成（跳过或验证完成）
    """
    st.subheader("🔍 验证标定")
    
    # 显示标定成功信息
    st.success(f"✅ 标定成功！使用了 {st.session_state.temp_num_points} 个点，标定误差: {st.session_state.temp_calibration_error:.2f} 像素")
    
    st.markdown("---")
    st.markdown("### 是否需要验证标定精度？")
    st.info("💡 您可以选择一条车道的左右边缘两点，系统会计算车道宽度来验证标定精度。标准车道宽度约为 **3.75米**。")
    
    # 初始化验证状态
    if 'validation_step' not in st.session_state:
        st.session_state.validation_step = 'ask'  # 'ask', 'selecting', 'done'
    if 'val_left_point' not in st.session_state:
        st.session_state.val_left_point = None
    if 'val_right_point' not in st.session_state:
        st.session_state.val_right_point = None
    if 'val_selecting' not in st.session_state:
        st.session_state.val_selecting = None  # 'left' or 'right'
    
    # 询问是否验证
    if st.session_state.validation_step == 'ask':
        col1, col2 = st.columns(2)
        
        if col1.button("✅ 进行验证", type="primary", key="do_validation"):
            st.session_state.validation_step = 'selecting'
            st.session_state.val_selecting = 'left'
            st.rerun()
        
        if col2.button("⏭️ 跳过验证", key="skip_validation"):
            st.session_state.calibration_done = True
            st.session_state.calibration_step = 'done'
            return True
    
    # 选择验证点
    elif st.session_state.validation_step == 'selecting':
        # 获取原始图像尺寸
        orig_height, orig_width = first_frame.shape[:2]
        
        # 在图像上绘制验证点
        display_image = first_frame.copy()
        
        # 绘制左边缘点（蓝色）
        if st.session_state.val_left_point:
            px, py = st.session_state.val_left_point
            cv2.circle(display_image, (px, py), 10, (255, 100, 0), -1)
            cv2.circle(display_image, (px, py), 12, (255, 255, 255), 2)
            cv2.putText(display_image, "L", (px + 14, py + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制右边缘点（红色）
        if st.session_state.val_right_point:
            px, py = st.session_state.val_right_point
            cv2.circle(display_image, (px, py), 10, (0, 100, 255), -1)
            cv2.circle(display_image, (px, py), 12, (255, 255, 255), 2)
            cv2.putText(display_image, "R", (px + 14, py + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 如果两点都有，绘制连线
        if st.session_state.val_left_point and st.session_state.val_right_point:
            cv2.line(display_image, st.session_state.val_left_point, 
                    st.session_state.val_right_point, (0, 255, 255), 2)
        
        # 转换显示
        display_image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_image_rgb)
        
        max_display_width = 800
        scale = min(max_display_width / orig_width, 1.0)
        display_width = int(orig_width * scale)
        display_height = int(orig_height * scale)
        
        # 显示当前选择状态
        if st.session_state.val_selecting == 'left':
            st.markdown("### 📍 请点击图片选择 **车道左边缘** 点")
            st.info("🔵 点击图片选择车道左边缘的一个点")
        elif st.session_state.val_selecting == 'right':
            st.markdown("### 📍 请点击图片选择 **车道右边缘** 点")
            st.info("🔴 点击图片选择车道右边缘的一个点")
        else:
            st.markdown("### 📍 验证点选择完成")
        
        # 点击图片
        clicked_coords = streamlit_image_coordinates(
            pil_image,
            width=display_width,
            height=display_height,
            key="validation_image"
        )
        
        # 处理点击
        if clicked_coords is not None:
            click_x = int(clicked_coords['x'] / scale)
            click_y = int(clicked_coords['y'] / scale)
            current_click = (click_x, click_y)
            
            # 检查是否是新点击
            last_val_click = st.session_state.get('last_val_click', None)
            if last_val_click != current_click:
                st.session_state.last_val_click = current_click
                
                if st.session_state.val_selecting == 'left':
                    st.session_state.val_left_point = current_click
                    st.session_state.val_selecting = 'right'
                    st.toast(f"✅ 左边缘点已选择: ({click_x}, {click_y})")
                    st.rerun()
                elif st.session_state.val_selecting == 'right':
                    st.session_state.val_right_point = current_click
                    st.session_state.val_selecting = None
                    st.toast(f"✅ 右边缘点已选择: ({click_x}, {click_y})")
                    st.rerun()
        
        # 显示当前选择的点
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.val_left_point:
                st.metric("🔵 左边缘点", f"({st.session_state.val_left_point[0]}, {st.session_state.val_left_point[1]})")
            else:
                st.metric("🔵 左边缘点", "未选择")
        with col2:
            if st.session_state.val_right_point:
                st.metric("🔴 右边缘点", f"({st.session_state.val_right_point[0]}, {st.session_state.val_right_point[1]})")
            else:
                st.metric("🔴 右边缘点", "未选择")
        
        # 操作按钮
        st.markdown("---")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        # 重新选择左边缘
        if btn_col1.button("🔄 重选左边缘", key="reselect_left"):
            st.session_state.val_selecting = 'left'
            st.session_state.last_val_click = None
            st.rerun()
        
        # 重新选择右边缘
        if btn_col2.button("🔄 重选右边缘", key="reselect_right"):
            st.session_state.val_selecting = 'right'
            st.session_state.last_val_click = None
            st.rerun()
        
        # 执行验证
        can_validate = st.session_state.val_left_point is not None and st.session_state.val_right_point is not None
        
        if btn_col3.button("✅ 执行验证", type="primary", key="run_validation", disabled=not can_validate):
            # 计算车道宽度
            width, status = speed_estimator.validate_lane_width(
                st.session_state.val_left_point,
                st.session_state.val_right_point
            )
            st.session_state.validation_result = (width, status)
            st.session_state.validation_step = 'done'
            st.rerun()
        
        # 跳过验证
        st.markdown("---")
        if st.button("⏭️ 跳过验证，直接开始检测", key="skip_validation_2"):
            st.session_state.calibration_done = True
            st.session_state.calibration_step = 'done'
            return True
    
    # 显示验证结果
    elif st.session_state.validation_step == 'done':
        if 'validation_result' in st.session_state:
            width, status = st.session_state.validation_result
            
            st.markdown("### 📊 验证结果")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("计算的车道宽度", f"{width:.2f} 米")
            with col2:
                st.metric("标准车道宽度", "3.75 米")
            
            st.markdown(f"**评估结果**: {status}")
        
        st.markdown("---")
        if st.button("✅ 完成，开始检测", type="primary", key="finish_validation"):
            st.session_state.calibration_done = True
            st.session_state.calibration_step = 'done'
            return True
    
    return False


def main():
    """主函数"""
    # 标题
    st.title("🚗 智能交通识别系统")
    st.markdown("---")
    
    # 初始化 session_state
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'calibration_done' not in st.session_state:
        st.session_state.calibration_done = False
    if 'processing_done' not in st.session_state:
        st.session_state.processing_done = False
        
    # ==================== 侧边栏 ====================
    with st.sidebar:
        st.header("控制面板")
        
        # 文件上传
        st.subheader("📁 文件上传")
        uploaded_file = st.file_uploader(
            "上传图片或视频",
            type=IMAGE_EXTENSIONS + VIDEO_EXTENSIONS,
            help="支持格式：JPG, PNG, MP4, AVI, MOV",
        )
        
        # 检测文件变化，重置状态
        if uploaded_file:
            if st.session_state.uploaded_file_name != uploaded_file.name:
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.calibration_done = False
                st.session_state.processing_done = False
                st.session_state.calibration_step = 'adding_points'
                # 重置标定相关状态
                if 'calib_points' in st.session_state:
                    st.session_state.calib_points = []
                if 'pending_click' in st.session_state:
                    st.session_state.pending_click = None
                if 'validation_step' in st.session_state:
                    st.session_state.validation_step = 'ask'
                if 'val_left_point' in st.session_state:
                    st.session_state.val_left_point = None
                if 'val_right_point' in st.session_state:
                    st.session_state.val_right_point = None
                
        # 功能选择
        st.subheader("🔧 功能选择")
        
        # 1. 车型识别 (默认不勾选)
        enable_type = st.checkbox("车型识别", value=False, key="enable_type")
        
        # 2. 车牌识别 (默认不勾选)
        enable_plate = st.checkbox("车牌识别", value=False, key="enable_plate")
        
        # 3. 车速识别 (默认不勾选，始终显示，但在非视频模式下禁用)
        # 判断当前文件状态
        is_video = False
        if uploaded_file:
            is_video = is_video_file(uploaded_file.name)
            
        # 禁用条件：已上传文件 且 不是视频
        speed_disabled = (uploaded_file is not None) and (not is_video)
        
        # 渲染复选框 (利用 disabled 参数控制是否可选)
        enable_speed = st.checkbox("车速识别", value=False, key="enable_speed", disabled=speed_disabled)
        
        # 额外的 UI 提示和逻辑安全锁
        if speed_disabled:
            st.caption("💡 图片模式不支持车速检测")
            enable_speed = False  # 强制设为 False，防止逻辑错误
                
        # 开始检测按钮
        st.subheader("🚀 操作")
        
        # 按钮启用条件
        can_start = uploaded_file is not None
        if enable_speed and not st.session_state.calibration_done:
            can_start = False
            st.warning("⚠️ 请先完成车速标定")
            
        start_button = st.button(
            "开始检测", 
            type="primary", 
            disabled=not can_start,
            key="start_detection"
        )
        
    # ==================== 主展示区 ====================
    
    # 默认状态
    if not uploaded_file:
        st.info("👋 欢迎使用智能交通识别系统！请在左侧上传图片或视频开始使用。")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🚙 车型识别")
            st.write("识别车辆类型（轿车、SUV、货车等）")
        with col2:
            st.markdown("### 🔢 车牌识别")
            st.write("检测并识别车牌号码")
        with col3:
            st.markdown("### ⚡ 车速识别")
            st.write("估算视频中车辆的行驶速度（仅视频）")
            
        return
        
    # 保存上传的文件
    temp_input_path = TEMP_DIR / uploaded_file.name
    with open(temp_input_path, 'wb') as f:
        f.write(uploaded_file.read())
        
    # 判断文件类型
    is_image = is_image_file(uploaded_file.name)
    is_video = is_video_file(uploaded_file.name)
    
    # ========== 图片处理 ==========
    if is_image:
        if start_button:
            with st.spinner("正在处理..."):
                # 读取图片
                image = cv2.imread(str(temp_input_path))
                
                if image is None:
                    st.error("无法读取图片文件")
                    return
                    
                # 处理图片
                vis_image, df = process_image(image, enable_plate, enable_type)
                
                st.session_state.processing_done = True
                st.session_state.result_image = vis_image
                st.session_state.result_df = df
                
        # 显示结果
        if st.session_state.processing_done and hasattr(st.session_state, 'result_image'):
            st.subheader("📷 检测结果")
            
            # 显示处理后的图片
            st.image(cv2.cvtColor(st.session_state.result_image, cv2.COLOR_BGR2RGB), 
                     caption="处理结果", use_container_width=True)
            
            # 新增：显示统计图表
            if hasattr(st.session_state, 'temp_vehicle_counts'):
                st.subheader("📊 数据统计")
                draw_statistics_charts(st.session_state.temp_vehicle_counts, None)
            
            # 显示统计表格
            if not st.session_state.result_df.empty:
                st.subheader("📊 检测统计")
                st.dataframe(st.session_state.result_df, use_container_width=True)
            else:
                st.info("未检测到车辆/车牌")
                
            # 下载按钮
            _, ext = os.path.splitext(uploaded_file.name)
            output_filename = f"result_{int(time.time())}{ext}"
            
            # 编码图片
            _, buffer = cv2.imencode(ext, st.session_state.result_image)
            
            st.download_button(
                label="📥 下载处理后的图片",
                data=buffer.tobytes(),
                file_name=output_filename,
                mime=f"image/{ext[1:]}"
            )
        else:
            # 显示原图预览
            st.subheader("📷 图片预览")
            image = cv2.imread(str(temp_input_path))
            if image is not None:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="原图", use_container_width=True)
                
    # ========== 视频处理 ==========
    elif is_video:
        # 初始化标定步骤状态
        if 'calibration_step' not in st.session_state:
            st.session_state.calibration_step = 'adding_points'  # 'adding_points', 'ask_validation', 'done'
        
        # 车速标定界面
        if enable_speed and not st.session_state.calibration_done:
            speed_estimator = load_speed_estimator()
            
            if speed_estimator:
                first_frame = speed_estimator.get_first_frame(str(temp_input_path))
                
                if first_frame is not None:
                    # 根据标定步骤显示不同界面
                    if st.session_state.calibration_step == 'adding_points':
                        calibration_interface(first_frame, speed_estimator)
                    elif st.session_state.calibration_step == 'ask_validation':
                        # 需要重新执行标定以获得 speed_estimator 的状态
                        pixel_points = [(pt['px'], pt['py']) for pt in st.session_state.calib_points]
                        world_points = [(pt['wx'], pt['wy']) for pt in st.session_state.calib_points]
                        speed_estimator.calibrate_from_points(pixel_points, world_points)
                        validation_interface(first_frame, speed_estimator)
                else:
                    st.error("无法读取视频第一帧")
            return
            
        # 处理视频
        if start_button:
            speed_estimator = load_speed_estimator() if enable_speed else None
            
            # 如果已标定，需要重新加载并设置标定参数
            if enable_speed and st.session_state.calibration_done:
                # 重新标定
                pixel_points = [(pt['px'], pt['py']) for pt in st.session_state.calib_points]
                world_points = [(pt['wx'], pt['wy']) for pt in st.session_state.calib_points]
                speed_estimator.calibrate_from_points(pixel_points, world_points)
                
            st.subheader("⏳ 正在处理视频...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"处理进度: {progress*100:.1f}%")
                
            try:
                output_path = process_video(
                    str(temp_input_path),
                    enable_plate=enable_plate,
                    enable_type=enable_type,
                    enable_speed=enable_speed,
                    speed_estimator=speed_estimator,
                    progress_callback=update_progress
                )
                
                st.session_state.processing_done = True
                st.session_state.result_video_path = output_path
                
                progress_bar.progress(1.0)
                status_text.text("处理完成！")
                
            except Exception as e:
                st.error(f"视频处理失败: {e}")
                return
                
        # 显示结果
        if st.session_state.processing_done and hasattr(st.session_state, 'result_video_path'):
            st.subheader("🎬 处理结果")
            
            # 显示视频
            with open(st.session_state.result_video_path, 'rb') as f:
                video_bytes = f.read()
                st.video(video_bytes)

            # 新增：显示统计图表 (饼图 + 折线图)
            if hasattr(st.session_state, 'temp_vehicle_counts') and hasattr(st.session_state, 'temp_time_series'):
                st.subheader("📊 交通数据分析")
                draw_statistics_charts(st.session_state.temp_vehicle_counts, 
                                    st.session_state.temp_time_series)
                
            # 下载按钮
            st.download_button(
                label="📥 下载处理后的视频",
                data=video_bytes,
                file_name=f"result_{int(time.time())}.mp4",
                mime="video/mp4"
            )
        else:
            # 显示原视频预览
            st.subheader("🎬 视频预览")
            st.video(str(temp_input_path))


if __name__ == "__main__":
    main()

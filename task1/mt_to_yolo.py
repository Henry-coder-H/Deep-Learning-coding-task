'''
本代码用于将 BIT-Vehicle 数据集转换为 YOLO 格式的数据集，方便使用 YOLO 模型进行车辆检测任务的训练和验证。
功能：
支持从 BIT-Vehicle 数据集的 .mat 文件中读取车辆信息
自动划分训练集、验证集和测试集 (比例 7:2:1)
将车辆的边界框坐标转换为 YOLO 格式
生成对应的 YOLO 标签文件（.txt)
生成 data.yaml 文件，用于 YOLO 模型训练时指定数据集路径和类别信息
'''
import scipy.io as sio
import os
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= 配置区域 =================

BASE_PATH = "/Deep-Learning-coding-task/task1/dataset/BIT_Vehicle_Dataset"  # 数据集根目录
MAT_FILE = os.path.join(BASE_PATH, "VehicleInfo.mat")  #  mat 文件路径
IMAGES_DIR = os.path.join(BASE_PATH, "images")         # 原始图片所在的文件夹路径
OUTPUT_DIR = "/data2/zhuangyn/Deep-Learning-coding-task/task1/dataset/BIT_YOLO_Dataset"   # 输出的 YOLO 格式数据集路径 (建议输出到当前目录或指定目录)

# 类别映射表 (BIT-Vehicle 的 6 个类别)
CLASS_MAP = {
    'Bus': 0,
    'Microbus': 1,
    'Minivan': 2,
    'Sedan': 3,
    'SUV': 4,
    'Truck': 5
}
# ===========================================

def convert_to_yolo_bbox(img_w, img_h, x1, y1, x2, y2):
    """将 left, top, right, bottom 转换为 YOLO x_center, y_center, w, h"""
    dw = 1.0 / img_w
    dh = 1.0 / img_h
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    
    # 归一化
    x_center *= dw
    y_center *= dh
    w *= dw
    h *= dh
    return x_center, y_center, w, h

def main():
    #  创建输出目录结构 
    if os.path.exists(OUTPUT_DIR):
        print(f"警告: 输出目录 {OUTPUT_DIR} 已存在，可能会覆盖文件。")
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    print(f"正在加载 {MAT_FILE} ...")
    try:
        mat_data = sio.loadmat(MAT_FILE)
    except FileNotFoundError:
        print(f"错误：找不到 {MAT_FILE}，请确认文件路径。")
        return

    # 获取核心数据 struct
    vehicle_info = mat_data['VehicleInfo']
    total_imgs = len(vehicle_info)
    print(f"检测到 {total_imgs} 张图像元数据。")

    #  数据集划分逻辑 (7:2:1) 
    
    # 第一步：分出 70% 训练集，剩余 30% (验证+测试)
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    
    # 第二步：将剩余的 30% 按照 2:1 的比例分为 验证集(20%) 和 测试集(10%)
    # 0.3 * (2/3) = 0.2 (Val), 0.3 * (1/3) = 0.1 (Test)
    val_idx, test_idx = train_test_split(temp_idx, test_size=1/3, random_state=42)
    
    # 转换为集合加速查找
    train_set = set(train_idx)
    val_set = set(val_idx)
    # 剩下的就是 test_set
    
    print(f"划分完成: 训练集 {len(train_idx)}, 验证集 {len(val_idx)}, 测试集 {len(test_idx)}")

    processed_count = 0
    
    for i in tqdm(range(total_imgs)):
        info = vehicle_info[i][0]
        
        # 提取文件名
        filename = info['name'][0]
        
        # 提取宽高
        h = int(info['height'][0][0])
        w = int(info['width'][0][0])
        
        # 构造源文件路径
        src_img_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(src_img_path):
            # 有时候数据集中可能缺少某些图片，跳过
            continue

        # 决定当前图片属于哪个集合
        if i in train_set:
            subset = 'train'
        elif i in val_set:
            subset = 'val'
        else:
            subset = 'test'
        
        # 目标路径
        dst_img_path = os.path.join(OUTPUT_DIR, 'images', subset, filename)
        # 将 .jpg 替换为 .txt
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        dst_label_path = os.path.join(OUTPUT_DIR, 'labels', subset, txt_filename)

        # 复制图片
        shutil.copy2(src_img_path, dst_img_path)

        # 处理标签
        yolo_lines = []
        
        # 提取 vehicles 结构体
        n_vehicles = int(info['nVehicles'][0][0])
        
        if n_vehicles > 0:
            vehicles_struct = info['vehicles']
            
            if vehicles_struct.size > 0:
                # 针对每辆车
                for vehicle in vehicles_struct.flat:
                    try:
                        left = int(vehicle['left'][0][0])
                        top = int(vehicle['top'][0][0])
                        right = int(vehicle['right'][0][0])
                        bottom = int(vehicle['bottom'][0][0])
                        
                        category = vehicle['category'][0]
                        
                        if category in CLASS_MAP:
                            class_id = CLASS_MAP[category]
                            
                            xc, yc, norm_w, norm_h = convert_to_yolo_bbox(w, h, left, top, right, bottom)
                            
                            # 坐标越界检查
                            xc = max(0, min(1, xc))
                            yc = max(0, min(1, yc))
                            norm_w = max(0, min(1, norm_w))
                            norm_h = max(0, min(1, norm_h))
                            
                            yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")
                    except Exception as e:
                        pass

        # 写入 txt 文件 (即使没有车，也要生成空txt文件作为负样本)
        with open(dst_label_path, 'w') as f:
            if yolo_lines:
                f.write('\n'.join(yolo_lines))
            
        processed_count += 1

    # 生成 data.yaml
    generate_yaml()
    print(f"\n转换完成！数据集保存在: {os.path.abspath(OUTPUT_DIR)}")
    print("你可以直接用 yolo detect train data=BIT_YOLO_Dataset/data.yaml 进行训练了")

def generate_yaml():
    # 注意：YOLOv8/v5/v11 训练时 data.yaml 中的路径通常建议使用绝对路径
    abs_path = os.path.abspath(OUTPUT_DIR)
    
    yaml_content = f"""
path: {abs_path}
train: images/train
val: images/val
test: images/test 

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.keys())}
"""
    with open(os.path.join(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

if __name__ == "__main__":
    main()
'''把 YOLO 格式的 BIT-Vehicle 检测数据集 一键裁剪成 分类数据集，每辆车单独保存为一张图片，并按类别归档，可直接用于 ResNet / EfficientNet / Swin 等分类模型训练。'
BIT_YOLO_Dataset/        ← 输入(YOLO 格式）
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/

BIT_CLS_Dataset/         ← 输出（分类格式）
├── train/
│   ├── Bus/
│   ├── Microbus/
│   ├── Minivan/
│   ├── Sedan/
│   ├── SUV/
│   └── Truck/
└── val/
    ├── Bus/
    ├── Microbus/
    ├── Minivan/
    ├── Sedan/
    ├── SUV/
    └── Truck/
└── test/
    ├── Bus/
    ├── Microbus/
    ├── Minivan/
    ├── Sedan/
    ├── SUV/
    └── Truck/
'''
import os
import cv2
import shutil
from tqdm import tqdm

# 输入：生成的 YOLO 数据集路径
YOLO_DATA_DIR = "/Deep-Learning-coding-task/task1/dataset/BIT_YOLO_Dataset"
# 输出：分类数据集路径
CLS_DATA_DIR = "/Deep-Learning-coding-task/task1/dataset/BIT_CLS_Dataset"

# 必须和 data.yaml 里的 names 顺序完全一致！
CLASS_NAMES = ['Bus', 'Microbus', 'Minivan', 'Sedan', 'SUV', 'Truck']
# =======================================

def yolo_to_bbox(x_center, y_center, w, h, img_w, img_h):
    """YOLO 归一化坐标转回像素坐标"""
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)

def main():
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(YOLO_DATA_DIR, 'images', split)
        label_dir = os.path.join(YOLO_DATA_DIR, 'labels', split)
        
        if not os.path.exists(img_dir):
            print(f"跳过 {split} 集：源文件夹不存在")
            continue

        output_root = os.path.join(CLS_DATA_DIR, split)
        
        # 创建类别文件夹
        for name in CLASS_NAMES:
            os.makedirs(os.path.join(output_root, name), exist_ok=True)
            
        print(f"正在处理 {split} 集...")
        
        # 遍历图片
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        
        count = 0
        for img_file in tqdm(img_files):
            # 读取图片
            img_path = os.path.join(img_dir, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            h_img, w_img = img.shape[:2]
            
            # 读取对应的 txt
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)
            
            if not os.path.exists(label_path):
                continue
                
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                try:
                    cls_id = int(parts[0])
                    
                    # 解析坐标
                    bbox = list(map(float, parts[1:]))
                    x1, y1, x2, y2 = yolo_to_bbox(*bbox, w_img, h_img)
                    
                    # 裁剪车辆
                    crop = img[y1:y2, x1:x2]
                    
                    # 过滤太小的碎片（比如小于 16x16 的）
                    if crop.shape[0] < 16 or crop.shape[1] < 16:
                        continue
                    
                    # 保存
                    if 0 <= cls_id < len(CLASS_NAMES):
                        cls_name = CLASS_NAMES[cls_id]
                        save_name = f"{os.path.splitext(img_file)[0]}_{idx}.jpg"
                        save_path = os.path.join(output_root, cls_name, save_name)
                        cv2.imwrite(save_path, crop)
                        count += 1
                except ValueError:
                    continue
        
        print(f"{split} 集处理完成，共生成 {count} 张分类图片。")

    print(f"\n 所有分类数据集构建完成！路径: {CLS_DATA_DIR}")

if __name__ == '__main__':
    main()
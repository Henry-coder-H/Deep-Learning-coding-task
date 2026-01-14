import cv2
from ultralytics import YOLO
import os

# ================= 配置区域 =================
# 数据集根目录和测试集分割文件
DATASET_ROOT = r"CCPD2019/CCPD2019"
TEST_SPLIT_FILE = r"CCPD2019/CCPD2019/splits/all_test.txt"
MODEL_PATH = 'weights/license_plate_detector.pt'

# 测试模式（True: 测试所有图片，False: 只测试大角度倾斜）
TEST_ALL_IMAGES = True

# 倾斜角度阈值（大于此值才显示）
TILT_THRESHOLD = 30  # 度数

# 随机采样开关（True: 随机选500张，False: 测试全部）
RANDOM_SAMPLE = True
SAMPLE_SIZE = 1000

# IoU阈值（用于判断检测是否正确）
IOU_THRESHOLD = 0.5

# ================= CCPD 文件名解析 =================
def parse_ccpd_tilt(filename):
    """从 CCPD 文件名中提取倾斜角度（以0°为基准，返回偏离角度）"""
    try:
        # 格式示例: 01-90_94-...jpg
        # 第1部分（用-分隔）: 模糊度_水平倾斜_垂直倾斜
        parts = filename.split('-')
        tilt_info = parts[1].split('_')
        horizontal = int(tilt_info[0])
        vertical = int(tilt_info[1])
        
        # 计算偏离角度（以90°为基准）
        horizontal_tilt = abs(horizontal)
        vertical_tilt = abs(vertical)
        max_tilt = max(horizontal_tilt, vertical_tilt)
        
        return max_tilt, horizontal_tilt, vertical_tilt
    except Exception as e:
        return None, None, None

def parse_ccpd_bbox(filename):
    """从 CCPD 文件名中提取真值边界框坐标"""
    try:
        # 格式: 面积-倾斜-边界框-四点-车牌-...
        # 第2部分是边界框: x1&y1_x2&y2 (注意使用&符号)
        parts = filename.split('-')
        bbox_str = parts[2]
        
        # 分割两个点: x1&y1 和 x2&y2
        points = bbox_str.split('_')
        pt1 = points[0].split('&')
        pt2 = points[1].split('&')
        
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        
        return x1, y1, x2, y2
    except Exception as e:
        return None, None, None, None

def calculate_iou(box1, box2):
    """计算两个边界框的IoU (Intersection over Union)
    box1, box2: (x1, y1, x2, y2)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # 交集面积
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height
    
    # 各自面积
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # 并集面积
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

# ================= 主程序 =================
# 1. 加载模型
if not os.path.exists(MODEL_PATH):
    print(f"❌ 错误：找不到 {MODEL_PATH}，请确认文件路径！")
    exit()

print(f"🚀 正在加载模型：{MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# 2. 获取测试图片列表
print(f"📂 正在读取测试集分割文件: {TEST_SPLIT_FILE}")
if not os.path.exists(TEST_SPLIT_FILE):
    print("❌ 测试集分割文件不存在！")
    exit()

with open(TEST_SPLIT_FILE, 'r') as f:
    image_paths = [line.strip() for line in f.readlines() if line.strip()]

print(f"📊 测试集包含 {len(image_paths)} 张图片")

# 随机采样逻辑
if RANDOM_SAMPLE and len(image_paths) > SAMPLE_SIZE:
    import random
    random.seed(42)  # 固定随机种子，保证结果可复现
    image_paths = random.sample(image_paths, SAMPLE_SIZE)
    print(f"📊 随机采样 {SAMPLE_SIZE} 张进行测试\n")
else:
    print()

# 根据模式选择测试图片
if TEST_ALL_IMAGES:
    # 测试所有图片
    test_images = [(img_rel_path, *parse_ccpd_tilt(os.path.basename(img_rel_path))) for img_rel_path in image_paths]
    test_images = [(img, mt, ht, vt) for img, mt, ht, vt in test_images if mt is not None]
    print(f"🔍 模式: 测试所有图片 ({len(test_images)} 张)")
else:
    # 筛选大角度倾斜的图片
    large_tilt_images = []
    for img_rel_path in image_paths:
        max_tilt, h_tilt, v_tilt = parse_ccpd_tilt(os.path.basename(img_rel_path))
        if max_tilt is not None and max_tilt > TILT_THRESHOLD:
            large_tilt_images.append((img_rel_path, max_tilt, h_tilt, v_tilt))
    
    # 按倾斜角度从大到小排序
    large_tilt_images.sort(key=lambda x: x[1], reverse=True)
    test_images = large_tilt_images
    print(f"🔍 模式: 只测试大角度倾斜 (倾斜>{TILT_THRESHOLD}°, {len(test_images)} 张)")

print("=" * 60)

# 3. 批量测试
detected_count = 0
correct_detection_count = 0  # IoU > 阈值的检测数
iou_list = []

for idx, (img_rel_path, max_tilt, h_tilt, v_tilt) in enumerate(test_images, 1):
    img_path = os.path.join(DATASET_ROOT, img_rel_path)
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"❌ [{idx}] 无法读取: {img_rel_path}")
        continue
    
    # 获取真值边界框
    gt_x1, gt_y1, gt_x2, gt_y2 = parse_ccpd_bbox(os.path.basename(img_rel_path))
    if gt_x1 is None:
        print(f"⚠️ [{idx}] 无法解析边界框: {img_rel_path}")
        continue
    
    # 进行推理
    results = model(image, conf=0.25, verbose=False)
    
    found_plate = False
    iou = 0.0
    
    for result in results:
        if len(result.boxes) > 0:
            found_plate = True
            detected_count += 1
            
            box = result.boxes[0]
            conf = float(box.conf[0])
            
            # 获取检测框坐标
            det_x1, det_y1, det_x2, det_y2 = map(int, box.xyxy[0])
            
            # 计算IoU
            iou = calculate_iou(
                (det_x1, det_y1, det_x2, det_y2),
                (gt_x1, gt_y1, gt_x2, gt_y2)
            )
            iou_list.append(iou)
            
            # 判断是否为正确检测
            if iou >= IOU_THRESHOLD:
                correct_detection_count += 1
            
            break
    
    if not found_plate:
        iou_list.append(0.0)
    
    # 每100张打印一次进度
    if idx % 100 == 0:
        current_detect_rate = (detected_count / idx) * 100
        current_acc_rate = (correct_detection_count / idx) * 100
        print(f"🚀 进度: [{idx}/{len(test_images)}] | "
              f"检测率: {current_detect_rate:.1f}% | "
              f"定位准确率: {current_acc_rate:.1f}%")

# 4. 输出统计
print("\n" + "="*60)
print(f"📊 车牌检测评估报告")
print("="*60)
print(f"📂 测试集: {DATASET_ROOT}")
if TEST_ALL_IMAGES:
    print(f"🔍 测试模式: 所有图片")
else:
    print(f"🔍 测试模式: 大角度倾斜 (>{TILT_THRESHOLD}°)")
print(f"📏 IoU阈值: > {IOU_THRESHOLD}")
print("-"*60)
print(f"1️⃣  检测统计")
print(f"   - 测试图片总数: {len(test_images)} 张")
print(f"   - 成功检测到车牌: {detected_count} 张")
print(f"   - 检测率: {(detected_count/len(test_images)*100):.1f}%")
print("-"*60)
print(f"2️⃣  定位准确率 (基于IoU)")
print(f"   - 准确检测数 (IoU>{IOU_THRESHOLD}): {correct_detection_count} 张")
print(f"   - 定位准确率: {(correct_detection_count/len(test_images)*100):.1f}%")
if iou_list:
    avg_iou = sum(iou_list) / len(iou_list)
    print(f"   - 平均IoU: {avg_iou:.3f}")
print("=" * 60)
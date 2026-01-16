import sys
import os
import cv2
import torch
import numpy as np
import time
from ultralytics import YOLO
import psutil

# ================= 配置区域 =================
DATASET_ROOT = r"CCPD2019/CCPD2019"
TEST_SPLIT_FILE = r"CCPD2019/CCPD2019/splits/all_test.txt"
HARDTEST_SPLIT_FILE = r"CCPD2019/CCPD2019/splits/all_hardtest.txt"

YOLO_WEIGHTS = 'weights/license_plate_detector.pt'
LPR_WEIGHTS = 'weights/lprnet_best.pth'

# 随机采样开关（True: 随机选500张，False: 测试全部）
RANDOM_SAMPLE = False
SAMPLE_SIZE = 500

# 引入 LPRNet
current_dir = os.path.dirname(os.path.abspath(__file__))
lprnet_path = os.path.join(current_dir, 'LPRNet_Pytorch')
if lprnet_path not in sys.path:
    sys.path.append(lprnet_path)
    
from model.LPRNet import LPRNet

# ================= CCPD 数据集解析标准 =================
CCPD_PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
CCPD_ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# ================= LPRNet 字符表 =================
CHARS = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', "青", "宁", "新", "警", "学", 
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']

def parse_ccpd_filename(filename):
    """从 CCPD 文件名中提取正确车牌号 (Ground Truth)"""
    try:
        base_name = os.path.basename(filename)
        parts = base_name.split('-')
        
        # 第4部分: 车牌索引
        label_str = parts[4] 
        idxs = label_str.split('_')
        
        # 映射转换
        province = CCPD_PROVINCES[int(idxs[0])]
        rest = [CCPD_ADS[int(i)] for i in idxs[1:]]
        
        plate_number = province + "".join(rest)
        return plate_number
    except Exception as e:
        return None

def load_lprnet(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lprnet = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load(weights_path, map_location=device))
    lprnet.eval()
    return lprnet, device

def decode_lpr_output(preds):
    preds = preds.cpu().detach().numpy()
    label_indices = np.argmax(preds, axis=1)
    decoded_str = ""
    last_char = -1
    for idx in label_indices[0]:
        if idx != last_char and idx != len(CHARS) - 1:
            decoded_str += CHARS[idx]
        last_char = idx
    return decoded_str

def preprocessing_lpr(img, device):
    img = cv2.resize(img, (94, 24))
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    return img

def main():
    print("=" * 60)
    print("📊 车牌识别模型性能评估")
    print("=" * 60)
    
    # --- 读取测试集和hard test集 ---
    print(f"\n📂 正在读取测试集: {TEST_SPLIT_FILE}")
    with open(TEST_SPLIT_FILE, 'r') as f:
        test_paths = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"📂 正在读取鲁棒性测试集: {HARDTEST_SPLIT_FILE}")
    with open(HARDTEST_SPLIT_FILE, 'r') as f:
        hardtest_paths = set([line.strip() for line in f.readlines() if line.strip()])
    
    total_test_images = len(test_paths)
    print(f"✅ 测试集加载完成: {total_test_images} 张图片")
    print(f"✅ 鲁棒性子集: {len(hardtest_paths)} 张图片\n")
    
    # --- 随机采样逻辑 ---
    if RANDOM_SAMPLE and len(test_paths) > SAMPLE_SIZE:
        import random
        random.seed(42)  # 固定随机种子，保证结果可复现
        test_paths = random.sample(test_paths, SAMPLE_SIZE)
        total_test_images = len(test_paths)
        print(f"📊 随机采样 {SAMPLE_SIZE} 张进行测试...\n")

    # --- 加载模型 ---
    print("🤖 正在加载模型...")
    yolo_model = YOLO(YOLO_WEIGHTS)
    lpr_model, device = load_lprnet(LPR_WEIGHTS)
    print(f"✅ 模型加载完成 (设备: {device})\n")

    # --- 统计计数器 ---
    # 全局统计
    count_full_match = 0          # 整牌全对的数量
    count_char_correct = 0        # 字符认对的总数
    count_char_total = 0          # 字符总数
    inference_times = []          # 推理耗时
    
    # 鲁棒性统计（all_hardtest 子集）
    hardtest_count = 0                  # 参与统计的hard test样本数
    hardtest_full_match = 0             # hard test全对数量

    # --- 开始评估 ---
    print("🚀 开始批量评估...\n")
    time_start = time.time()
    
    for i, img_rel_path in enumerate(test_paths):
        # 判断是否属于hardtest子集
        is_hardtest = img_rel_path in hardtest_paths
        
        img_path = os.path.join(DATASET_ROOT, img_rel_path)
        
        # 1. 解析真值
        ground_truth = parse_ccpd_filename(img_rel_path)
        if not ground_truth:
            continue
        
        # 2. 读取图片
        img = cv2.imread(img_path)
        if img is None:
            continue

        # --- 推理计时开始 ---
        t0 = time.time()

        # 3. YOLO 检测
        results = yolo_model(img, verbose=False)
        
        detected_text = None
        
        # 寻找车牌
        for result in results:
            if len(result.boxes) > 0:
                box = result.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 裁剪
                h, w = img.shape[:2]
                pad = 3
                crop_y1, crop_y2 = max(0, y1-pad), min(h, y2+pad)
                crop_x1, crop_x2 = max(0, x1-pad), min(w, x2+pad)
                plate_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # LPRNet 识别
                input_tensor = preprocessing_lpr(plate_crop, device)
                with torch.no_grad():
                    preds = lpr_model(input_tensor)
                    detected_text = decode_lpr_output(preds)
                break
        
        # --- 推理计时结束 ---
        t_cost = (time.time() - t0) * 1000  # ms
        inference_times.append(t_cost)

        # 4. 统计结果
        is_correct = False
        if detected_text:
            # 全字匹配
            if detected_text == ground_truth:
                count_full_match += 1
                is_correct = True
            
            # 字符级匹配
            length = min(len(detected_text), len(ground_truth))
            for j in range(length):
                if detected_text[j] == ground_truth[j]:
                    count_char_correct += 1
        
        # 累计字符总数
        count_char_total += len(ground_truth)
        
        # 鲁棒性统计
        if is_hardtest:
            hardtest_count += 1
            if is_correct:
                hardtest_full_match += 1

        # 打印进度
        if (i + 1) % 100 == 0:
            print(f"   处理进度: {i+1}/{total_test_images} ({(i+1)/total_test_images*100:.1f}%)")

    # --- 计算指标 ---
    total_time = time.time() - time_start
    avg_latency = np.mean(inference_times) if inference_times else 0
    fps = 1000 / avg_latency if avg_latency > 0 else 0
    
    # 准确率
    full_match_acc = (count_full_match / total_test_images) * 100
    char_acc = (count_char_correct / count_char_total) * 100 if count_char_total > 0 else 0
    
    # 鲁棒性
    hardtest_acc = (hardtest_full_match / hardtest_count) * 100 if hardtest_count > 0 else 0

    # --- 输出结果 ---
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"评测图片数量: {total_test_images}")
    print()
    print("1. 准确率 (Accuracy)")
    print(f"   - 全字匹配率（整牌全对）: {full_match_acc:.2f}%")
    print(f"   - 字符准确率（逐字符平均）: {char_acc:.2f}%")
    print()
    print("2. 推理速度 (Latency)")
    print(f"   - 平均单张耗时: {avg_latency:.2f} ms")
    print(f"   - FPS（每秒处理张数）: {fps:.2f}")
    print()
    print("3. 鲁棒性 (Robustness)")
    print(f"   - all_hardtest 子集整牌匹配率: {hardtest_acc:.2f}%   (样本数={hardtest_count})")
    print("=" * 60)
    print(f"总耗时: {total_time:.2f} 秒")
    print("=" * 60)

if __name__ == "__main__":
    main()

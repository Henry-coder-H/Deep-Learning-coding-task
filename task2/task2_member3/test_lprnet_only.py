import sys
import os
import cv2
import torch
import numpy as np
import time

# ================= 配置区域 =================
# 数据集根目录和测试集分割文件
DATASET_ROOT = r"CCPD2019/CCPD2019"
TEST_SPLIT_FILE = r"CCPD2019/CCPD2019/splits/all_test.txt"

# LPRNet权重路径
# LPR_WEIGHTS = 'LPRNet_Pytorch/weights/Final_LPRNet_model.pth'
# LPR_WEIGHTS = 'weights/lprnet_epoch_3.pth'
LPR_WEIGHTS = 'weights/lprnet_best.pth'
# LPR_WEIGHTS = 'YOLOv5-LPRNet-Licence-Recognition/weights/lprnet_best.pth'

# 随机采样开关（True: 随机选500张，False: 测试全部）
RANDOM_SAMPLE = True
SAMPLE_SIZE = 1000

# 大角度筛选开关（True: 只测试大角度倾斜>30°，False: 测试所有）
LARGE_TILT_ONLY = False
TILT_THRESHOLD = 30  # 大角度阈值（度）

# ================= 引入 LPRNet =================
current_dir = os.path.dirname(os.path.abspath(__file__))
lprnet_path = os.path.join(current_dir, 'LPRNet_Pytorch')
if lprnet_path not in sys.path:
    sys.path.append(lprnet_path)
    
from model.LPRNet import LPRNet

# ================= CCPD 数据集解析 =================
CCPD_PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
CCPD_ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# ================= LPRNet 字符表 =================
# CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', 
#          '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', 
#          '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', 
#          '新', 
#          '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
#          'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 
#          'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
#          'W', 'X', 'Y', 'Z', 'I', 'O', '-'
# ]

# LPRNet 训练需要的全局字符表（去重并排序，保持 'O' 作为空白符在最后或者是特定的位置）
# 这里我们构建一个包含所有可能字符的列表
CHARS = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏',
          '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼',
            '川', '贵', '云', '藏', '陕', '甘', "青", "宁", "新", "警", "学", 
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
           'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'] # 最后加个 '-' 作为空白符(blank)

def parse_ccpd_filename(filename):
    """从 CCPD 文件名中提取车牌号和倾斜角度"""
    try:
        base_name = os.path.basename(filename)
        parts = base_name.split('-')

        # 倾斜角度
        tilt_info = parts[1].split('_')
        horizontal_tilt_raw = int(tilt_info[0])
        vertical_tilt_raw = int(tilt_info[1])

        # 规则：以 'ccpd_base' 和 'ccpd_green' 开头的相对 90° 为基准，
        # 其他路径以 0° 为基准。使用传入的相对路径判断第一段目录名。
        norm = filename.replace('\\', '/').lstrip('./')
        first_comp = norm.split('/')[0].lower() if norm else ''
        if first_comp in ('ccpd_base', 'ccpd_green'):
            baseline = 90
        else:
            baseline = 0

        horizontal_tilt = abs(horizontal_tilt_raw - baseline)
        vertical_tilt = abs(vertical_tilt_raw - baseline)
        max_tilt = max(horizontal_tilt, vertical_tilt)
        
        # 车牌号
        label_str = parts[4] 
        idxs = label_str.split('_')
        province = CCPD_PROVINCES[int(idxs[0])]
        rest = [CCPD_ADS[int(i)] for i in idxs[1:]]
        plate_number = province + "".join(rest)
        
        return plate_number, max_tilt
    except Exception as e:
        return None, None

def parse_ccpd_bbox(filename):
    """从 CCPD 文件名中提取真值边界框坐标"""
    try:
        base_name = os.path.basename(filename)
        parts = base_name.split('-')
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
    # --- 初始化 ---
    print(f"📂 正在读取测试集分割文件: {TEST_SPLIT_FILE}")
    
    if not os.path.exists(TEST_SPLIT_FILE):
        print("❌ 测试集分割文件不存在！")
        return
    
    with open(TEST_SPLIT_FILE, 'r') as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"📊 测试集包含 {len(image_paths)} 张图片")
    
    # 随机采样逻辑
    if RANDOM_SAMPLE and len(image_paths) > SAMPLE_SIZE:
        import random
        random.seed(42)
        image_paths = random.sample(image_paths, SAMPLE_SIZE)
        print(f"📊 随机采样 {SAMPLE_SIZE} 张进行测试...\n")
    else:
        print(f"🔢 开始批量测试...\n")
    
    # 大角度筛选逻辑
    if LARGE_TILT_ONLY:
        filtered_paths = []
        for img_rel_path in image_paths:
            _, max_tilt = parse_ccpd_filename(img_rel_path)
            if max_tilt is not None and max_tilt > TILT_THRESHOLD:
                filtered_paths.append(img_rel_path)
        image_paths = filtered_paths
        print(f"🔍 筛选大角度倾斜照片: {len(image_paths)} 张 (倾斜>{TILT_THRESHOLD}°)\n")
    
    total_files = len(image_paths)

    # 加载模型
    print(f"🚀 正在加载 LPRNet 模型: {LPR_WEIGHTS}...")
    lpr_model, device = load_lprnet(LPR_WEIGHTS)
    print(f"✅ 模型加载成功，设备: {device}\n")

    # --- 统计计数器 ---
    count_full_match = 0      # 整牌全对
    count_char_correct = 0    # 字符认对的总数
    count_char_total = 0      # 字符总数
    count_bbox_fail = 0       # 边界框解析失败
    count_crop_fail = 0       # 裁剪失败（边界框异常）
    
    # 大角度倾斜统计
    count_large_tilt = 0
    count_large_tilt_correct = 0
    
    time_start_total = time.time()
    inference_times = []

    print("=" * 60)
    print("开始测试 LPRNet（使用真值边界框裁剪）")
    print("=" * 60)

    # --- 开始循环 ---
    for i, img_rel_path in enumerate(image_paths):
        img_path = os.path.join(DATASET_ROOT, img_rel_path)
        
        # 1. 解析真值车牌号和倾斜角度
        ground_truth, max_tilt = parse_ccpd_filename(img_rel_path)
        if not ground_truth:
            continue
        
        is_large_tilt = max_tilt > 30

        # 2. 解析真值边界框
        x1, y1, x2, y2 = parse_ccpd_bbox(img_rel_path)
        if x1 is None:
            count_bbox_fail += 1
            continue
        
        # 3. 读取图片
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 4. 使用真值边界框裁剪车牌
        h, w = img.shape[:2]
        pad = 3
        crop_y1, crop_y2 = max(0, y1-pad), min(h, y2+pad)
        crop_x1, crop_x2 = max(0, x1-pad), min(w, x2+pad)
        
        plate_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if plate_crop.size == 0 or plate_crop.shape[0] < 5 or plate_crop.shape[1] < 5:
            count_crop_fail += 1
            continue

        # --- 计时开始 ---
        t0 = time.time()

        # 5. LPRNet 识别
        try:
            input_tensor = preprocessing_lpr(plate_crop, device)
            with torch.no_grad():
                preds = lpr_model(input_tensor)
                detected_text = decode_lpr_output(preds)
        except Exception as e:
            detected_text = None
        
        # --- 计时结束 ---
        t_cost = (time.time() - t0) * 1000
        inference_times.append(t_cost)

        # 6. 对比统计
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
        
        # 大角度倾斜统计
        if is_large_tilt:
            count_large_tilt += 1
            if is_correct:
                count_large_tilt_correct += 1

        # 累计字符总数
        count_char_total += len(ground_truth)

        # 打印进度
        if (i+1) % 50 == 0:
            print(f"🚀 进度: {i+1}/{total_files} | 识别: {detected_text} | 真值: {ground_truth} | {'✅' if is_correct else '❌'}")

    # --- 计算最终指标 ---
    total_time_sec = time.time() - time_start_total
    avg_latency = np.mean(inference_times) if inference_times else 0
    fps = 1000 / avg_latency if avg_latency > 0 else 0
    
    valid_samples = total_files - count_bbox_fail - count_crop_fail
    acc_full = (count_full_match / valid_samples) * 100 if valid_samples > 0 else 0
    acc_char = (count_char_correct / count_char_total) * 100 if count_char_total > 0 else 0
    acc_large_tilt = (count_large_tilt_correct / count_large_tilt) * 100 if count_large_tilt > 0 else 0

    # --- 输出报表 ---
    print("\n" + "="*60)
    print("📊 LPRNet 识别性能评估报告（使用真值边界框）")
    print("="*60)
    print(f"📂 测试集:      {TEST_SPLIT_FILE}")
    print(f"🔢 测试样本:    {total_files} 张")
    print(f"✅ 有效样本:    {valid_samples} 张")
    if count_bbox_fail > 0:
        print(f"⚠️ 边界框解析失败: {count_bbox_fail} 张")
    if count_crop_fail > 0:
        print(f"⚠️ 裁剪失败:    {count_crop_fail} 张")
    print(f"⏱️ 总耗时:      {total_time_sec:.2f} 秒")
    print("-" * 60)
    print("1️⃣  准确率指标 (Accuracy)")
    print(f"   - 全字匹配率 (Full Match):  {acc_full:.2f}%")
    print(f"   - 字符准确率 (Char Acc):    {acc_char:.2f}%")
    print("-" * 60)
    print("2️⃣  速度指标 (Latency)")
    print(f"   - 平均耗时 (Latency):       {avg_latency:.2f} ms")
    print(f"   - 帧率 (FPS):               {fps:.2f} FPS")
    print("-" * 60)
    print("3️⃣  鲁棒性指标 (Robustness)")
    print(f"   - 大角度倾斜样本数:         {count_large_tilt} 张 (倾斜>30°)")
    print(f"   - 大角度倾斜识别率:         {acc_large_tilt:.2f}%")
    print("="*60)
    print("💡 说明：本测试使用文件名中的真值边界框裁剪车牌，")
    print("   纯粹评估 LPRNet 识别能力，不受 YOLO 检测精度影响。")

if __name__ == "__main__":
    main()

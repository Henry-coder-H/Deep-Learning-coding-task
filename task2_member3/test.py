import sys
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ================= 配置区域 =================
# 1. 这里填入你的 CCPD 图片路径 (绝对路径或相对路径)
# 例如: "ccpd_sample/base/003654...jpg"
# TEST_IMG_PATH = r"ccpd_sample/base/03-81_102-254I444_508I568-501I523_252I579_278I483_527I427-0_0_11_21_26_24_32-181-46.jpg" 
TEST_IMG_PATH = r"images.jpg"

# 2. 权重文件路径
YOLO_WEIGHTS = 'weights/license_plate_detector.pt'
LPR_WEIGHTS = 'weights/lprnet_best.pth'

# 引入 LPRNet
current_dir = os.path.dirname(os.path.abspath(__file__))
lprnet_path = os.path.join(current_dir, 'LPRNet_Pytorch')
if lprnet_path not in sys.path:
    sys.path.append(lprnet_path)
    
from model.LPRNet import LPRNet

# 这里我们构建一个包含所有可能字符的列表
CHARS = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏',
          '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼',
            '川', '贵', '云', '藏', '陕', '甘', "青", "宁", "新", "警", "学", 
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
           'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'] # 最后加个 '-' 作为空白符(blank)

def load_lprnet(weights_path):
    """加载 LPRNet 模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 这里的 class_num=68 是 LPRNet 的标准配置
    # lpr_max_len: 车牌最大长度，通常为8（中国车牌标准）
    # phase: False 表示测试模式
    lprnet = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    
    if not os.path.exists(weights_path):
        print(f"❌ 错误：找不到 LPRNet 权重文件: {weights_path}")
        return None
        
    print(f"📥 正在加载 LPRNet 权重: {weights_path}")
    lprnet.load_state_dict(torch.load(weights_path, map_location=device))
    lprnet.eval()
    return lprnet

def decode_lpr_output(preds):
    """解码 LPRNet 的输出 (Greedy Decode)"""
    preds = preds.cpu().detach().numpy() # (1, 68, 18)
    label_indices = np.argmax(preds, axis=1) # (1, 18)
    
    decoded_str = ""
    last_char = -1
    
    for idx in label_indices[0]:
        # LPRNet 使用 CTC Loss，需要处理重复字符和空白符
        # len(CHARS)-1 通常是空白符 '-'
        if idx != last_char and idx != len(CHARS) - 1:
            decoded_str += CHARS[idx]
        last_char = idx
        
    return decoded_str

def preprocessing_lpr(img):
    """LPRNet 专用的预处理: Resize -> Normalize -> Transpose"""
    # 1. Resize 到 94x24
    img = cv2.resize(img, (94, 24))
    img = img.astype('float32')
    
    # 2. 归一化 (这是 LPRNet 官方仓库的预处理方式)
    img -= 127.5
    img *= 0.0078125
    
    # 3. 转换维度 (H, W, C) -> (C, H, W) -> (1, C, H, W)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    
    return img

def main():
    # --- 1. 准备模型 ---
    print("🚀 正在初始化系统...")
    
    # 加载 YOLO
    if not os.path.exists(YOLO_WEIGHTS):
        print(f"❌ 错误：找不到 YOLO 权重: {YOLO_WEIGHTS}")
        return
    yolo_detector = YOLO(YOLO_WEIGHTS)
    
    # 加载 LPRNet
    lpr_net = load_lprnet(LPR_WEIGHTS)
    if lpr_net is None: return

    # --- 2. 读取图片 ---
    if TEST_IMG_PATH == "在此处粘贴你的图片路径":
        print("⚠️ 请在代码第 11 行填入真实的图片路径！")
        return
        
    if not os.path.exists(TEST_IMG_PATH):
        print(f"❌ 无法找到图片: {TEST_IMG_PATH}")
        return
        
    full_img = cv2.imread(TEST_IMG_PATH)
    print(f"📸 已读取图片: {TEST_IMG_PATH}")

    # --- 3. 第一阶段：检测 (YOLO) ---
    results = yolo_detector(full_img, verbose=False)
    
    found_plate = False
    
    for result in results:
        for box in result.boxes:
            # 获取坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # 简单的过滤：太小的框不要
            if (x2-x1) < 30 or (y2-y1) < 10: continue
            
            found_plate = True
            print(f"✅ 检测到车牌区域: [{x1}, {y1}, {x2}, {y2}] (置信度: {conf:.2f})")
            
            # --- 4. 裁剪 + 预处理 ---
            # 稍微外扩一点点(padding)，识别效果更好
            h, w = full_img.shape[:2]
            pad = 2
            crop_y1, crop_y2 = max(0, y1-pad), min(h, y2+pad)
            crop_x1, crop_x2 = max(0, x1-pad), min(w, x2+pad)
            
            plate_img = full_img[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # 显示裁剪的小图看看
            cv2.imwrite('debug_current_crop.jpg', plate_img)
            
            # 转为 Tensor
            input_tensor = preprocessing_lpr(plate_img)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_tensor = input_tensor.to(device)
            
            # --- 5. 第二阶段：识别 (LPRNet) ---
            with torch.no_grad():
                preds = lpr_net(input_tensor)
                result_text = decode_lpr_output(preds)
                
            print(f"🎉 最终识别结果: 【 {result_text} 】")
            
            # --- 6. 简单的可视化 ---
            cv2.rectangle(full_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 注意：cv2.putText 不支持中文，这里只显示英文或拼音，或者在终端看结果
            cv2.putText(full_img, "Detected", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 只识别置信度最高的一个就退出，避免重复
            break 
        if found_plate: break

    if not found_plate:
        print("⚠️ 未检测到任何车牌！")
    else:
        cv2.imshow("Result", full_img)
        print("按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
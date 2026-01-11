import cv2
import torch
import numpy as np
import sys
import os
import time
import psutil  # <--- 新增库：用于获取内存占用
from collections import Counter, defaultdict
from ultralytics import YOLO

# ================= 1. 配置区域 =================
VIDEO_PATH = 'test_video.mp4'
YOLO_WEIGHTS = 'weights/license_plate_detector.pt'
# LPR_WEIGHTS = 'LPRNet_Pytorch/weights/Final_LPRNet_model.pth'
LPR_WEIGHTS = 'weights/lprnet_best.pth'

# 【真值白名单】 (填入视频里所有正确的车牌)
TRUE_PLATES = [
    "京GPL768",
    "京BF1144",
    "京M76967",
    "京B06498",
    "京L87802",
    "京J27373",
    "京KS0537",
    "京JZ9445",
    # ... 继续添加
]

# ================= 2. 核心逻辑 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
lprnet_path = os.path.join(current_dir, 'LPRNet_Pytorch')
if lprnet_path not in sys.path:
    sys.path.append(lprnet_path)
from model.LPRNet import LPRNet

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

class VideoEvaluator:
    def __init__(self):
        self.fps_history = []
        self.predictions = defaultdict(list)
        
    def add_record(self, track_id, text, inference_ms, frame_idx):
        self.predictions[track_id].append((frame_idx, text))
        self.fps_history.append(inference_ms)

    def calculate_max_consecutive(self, frame_indices):
        """计算最大连续帧数"""
        if not frame_indices: return 0
        sorted_frames = sorted(frame_indices)
        max_cons = 1
        current_cons = 1
        for i in range(1, len(sorted_frames)):
            if sorted_frames[i] == sorted_frames[i-1] + 1:
                current_cons += 1
            else:
                max_cons = max(max_cons, current_cons)
                current_cons = 1
        return max(max_cons, current_cons)

    def print_report(self, vram_peak, ram_usage):
        print("\n" + "="*95)
        print("🎬 视频处理性能评估报告 (包含显存/内存统计)")
        print("="*95)
        
        avg_ms = np.mean(self.fps_history) if self.fps_history else 0
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        
        # 1. 速度指标
        print(f"⏱️  平均速度 (Latency):")
        print(f"    - FPS: {fps:.2f}")
        print(f"    - 单帧耗时: {avg_ms:.2f} ms")
        print("-" * 95)
        
        # 2. 识别详情
        print(f"{'ID':<4} | {'识别结果 (Result)':<12} | {'帧数':<4} | {'最大连续':<8} | {'占比':<6} | {'判定'}")
        print("-" * 95)
        
        found_true_plates = set()
        total_frames_processed = 0
        total_correct_frames = 0
        
        for tid, data_list in sorted(self.predictions.items()):
            all_texts = [x[1] for x in data_list]
            all_indices = [x[0] for x in data_list]
            
            counter = Counter(all_texts)
            sorted_results = counter.most_common()
            
            total_id_frames = len(all_texts)
            total_frames_processed += total_id_frames
            
            first_row = True
            for text, count in sorted_results:
                current_indices = [idx for idx, t in zip(all_indices, all_texts) if t == text]
                max_cons = self.calculate_max_consecutive(current_indices)
                ratio = (count / total_id_frames) * 100
                
                if text in TRUE_PLATES:
                    status = "✅ 正确"
                    found_true_plates.add(text)
                    total_correct_frames += count
                else:
                    status = "❌ 未知"
                
                id_str = str(tid) if first_row else ""
                print(f"{id_str:<4} | {text:<12} | {count:<4} | {max_cons:<8} | {ratio:.1f}%  | {status}")
                first_row = False
            
            print("-" * 95)

        # 3. 统计摘要
        recall = (len(found_true_plates) / len(TRUE_PLATES)) * 100 if TRUE_PLATES else 0
        frame_acc = (total_correct_frames / total_frames_processed) * 100 if total_frames_processed else 0
        
        print(f"📈 统计摘要:")
        print(f"   1. 召回率 (Recall):   {recall:.2f}%  (白名单里的 {len(TRUE_PLATES)} 辆车，找到了 {len(found_true_plates)} 辆)")
        print(f"   2. 帧准确率 (Frame Acc): {frame_acc:.2f}% (所有处理帧中，有 {total_correct_frames} 帧是完全正确的)")
        print("-" * 95)

        # 4. 显存/内存占用 (新增板块)
        print(f"💾 资源占用 (Memory Usage):")
        print(f"   - GPU 显存峰值 (VRAM Peak): {vram_peak:.2f} MB  (越小越好，防止 OOM)")
        print(f"   - CPU 内存占用 (RAM Usage): {ram_usage:.2f} MB")
        print("="*95)

def load_lprnet(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lprnet = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load(weights_path, map_location=device))
    lprnet.eval()
    return lprnet, device

def decode_lpr(preds):
    preds = preds.cpu().detach().numpy()
    label_indices = np.argmax(preds, axis=1)
    decoded_str = ""
    last_char = -1
    for idx in label_indices[0]:
        if idx != last_char and idx != len(CHARS) - 1:
            decoded_str += CHARS[idx]
        last_char = idx
    return decoded_str

def preprocess_plate(img, device):
    img = cv2.resize(img, (94, 24))
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).to(device)
    return img

def main():
    # --- 🔍 硬件自检 ---
    print(f"🖥️  正在检测硬件环境...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        # 重置显存统计，确保从当前脚本开始计算
        torch.cuda.reset_peak_memory_stats()
        print(f"✅ 成功调用 GPU: {gpu_name}")
    else:
        print("⚠️  未检测到 GPU，正在使用 CPU 跑代码")
    # ---------------------
    
    yolo = YOLO(YOLO_WEIGHTS)
    lpr, device = load_lprnet(LPR_WEIGHTS)
    evaluator = VideoEvaluator()
    cap = cv2.VideoCapture(VIDEO_PATH)
    cv2.namedWindow('Member 3 - Visualization', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Member 3 - Visualization', 1024, 768)

    print("🎥 开始运行 (V6 - 含内存统计)...")
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        t0 = time.time()
        
        # 追踪
        results = yolo.track(frame, persist=True, verbose=False, imgsz=320)
        
        for result in results:
            if result.boxes is None or result.boxes.id is None: continue
            boxes = result.boxes.xyxy.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                if (x2-x1) < 30: continue
                
                # 裁剪
                h, w = frame.shape[:2]
                crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                if crop.size == 0: continue
                
                # 识别
                inp = preprocess_plate(crop, device)
                with torch.no_grad():
                    text = decode_lpr(lpr(inp))
                
                t_cost = (time.time() - t0) * 1000
                evaluator.add_record(track_id, text, t_cost, frame_idx)
                
                # 可视化
                if text in TRUE_PLATES:
                    color = (0, 255, 0)
                    status_text = "MATCH"
                else:
                    color = (0, 0, 255)
                    status_text = "UNKNOWN"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # (绘图部分简化以保持流畅)

        cv2.imshow('Member 3 - Visualization', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # --- 📊 采集最终内存数据 ---
    vram_peak_mb = 0
    if torch.cuda.is_available():
        # 获取最大显存占用 (Max Memory Allocated)
        vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    # 获取当前进程的 RAM 占用
    process = psutil.Process(os.getpid())
    ram_usage_mb = process.memory_info().rss / (1024 ** 2)
    
    # 打印报表
    evaluator.print_report(vram_peak_mb, ram_usage_mb)

if __name__ == "__main__":
    main()
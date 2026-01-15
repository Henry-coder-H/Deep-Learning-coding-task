import torch
from ultralytics import YOLO
import cv2
import os
import argparse
import time
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt  # 新增：用于绘图
import pandas as pd             # 新增：用于数据管理

# 忽略警告
warnings.filterwarnings("ignore")

# ================= 🔧 全局配置 =================
COLORS = {
    'Bus': (0, 128, 255), 'Microbus': (0, 255, 255), 'Minivan': (255, 0, 255),
    'Sedan': (0, 255, 0), 'SUV': (255, 0, 0), 'Truck': (0, 0, 255)
}
DEFAULT_COLOR = (255, 255, 255)

class InferenceEngine:
    def __init__(self, model_path):
        print(f"👉 [初始化] 加载 YOLO 模型: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        # --- 📊 统计相关数据结构 ---
        self.vehicle_counts = defaultdict(set) # 记录不重复的 ID
        self.time_series_data = []             # 记录 (时间点, 实时车辆总数)

    def run(self, input_path, output_path, report_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            self._process_image(input_path, output_path, report_path)
        elif ext in ['.mp4', '.avi', '.mov']:
            self._process_video(input_path, output_path, report_path)

    def _process_image(self, img_path, save_path, report_path=None):
        """
        处理单张图片：推理 -> 统计 -> 绘图 -> 保存 -> 生成报告
        """
        print(f"🖼️ [图片] 开始处理: {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 错误: 无法读取图片 {img_path}")
            return

        # 1. 推理 (图片无需跟踪模式，使用 predict 即可)
        results = self.model.predict(img, conf=0.25, verbose=False)[0]

        # 2. 统计逻辑适配
        # 图片模式下没有 Track ID，为了适配 self.vehicle_counts 的 set 结构，
        # 我们使用当前帧的检测框索引(index)作为"伪ID"进行计数。
        if results.boxes:
            cls_ids = results.boxes.cls.int().cpu().numpy()
            for i, c_id in enumerate(cls_ids):
                class_name = self.class_names[c_id]
                # 使用 i 作为临时唯一标识，确保 len(set) 统计正确
                self.vehicle_counts[class_name].add(i)

        # 3. 绘图 (复用现有方法)
        self._draw_results(img, results, is_video=False)
        self._draw_statistics_panel(img)

        # 4. 智能修正保存路径后缀
        # 如果主程序传入的是 .mp4 后缀（针对视频的默认设置），强制改为 .jpg
        root, ext = os.path.splitext(save_path)
        if ext.lower() not in ['.jpg', '.jpeg', '.png']:
            save_path = root + ".jpg"
        
        cv2.imwrite(save_path, img)
        print(f"✅ 图片推理完成，已保存至: {save_path}")

        # 5. 生成分析报告 (如果传入了 report_path)
        if report_path:
            self._generate_report(report_path)

    def _process_video(self, vid_path, save_path, report_path):
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        frame_idx = 0
        print(f"🎥 [视频] 开始推理与数据挖掘...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 使用跟踪模式
            # results = self.model.track(frame, persist=True, conf=0.25, verbose=False)[0]
            results = self.model.predict(frame, conf=0.25, verbose=False)[0]
            
            # 1. 实时统计 ID
            if results.boxes.id is not None:
                track_ids = results.boxes.id.int().cpu().numpy()
                cls_ids = results.boxes.cls.int().cpu().numpy()
                for t_id, c_id in zip(track_ids, cls_ids):
                    self.vehicle_counts[self.class_names[c_id]].add(t_id)

            # 2. 采样：每秒记录一次车流量数据 (用于折线图)
            if frame_idx % int(fps) == 0:
                current_total = sum(len(ids) for ids in self.vehicle_counts.values())
                timestamp = frame_idx / fps
                self.time_series_data.append({'time': timestamp, 'count': current_total})

            # 3. 绘图与面板显示
            self._draw_results(frame, results, is_video=True)
            self._draw_statistics_panel(frame)
            
            out.write(frame)
            frame_idx += 1
            if frame_idx % 30 == 0: print(f"Processing... {frame_idx} frames", end='\r')

        cap.release()
        out.release()
        
        # 🚀 任务结束：生成二次数据挖掘报告
        self._generate_report(report_path)

    def _draw_results(self, img, results, is_video=False):
        if results.boxes:
            boxes = results.boxes.xyxy.cpu().numpy()
            clses = results.boxes.cls.int().cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            ids = results.boxes.id.int().cpu().numpy() if (is_video and results.boxes.id is not None) else None
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.class_names[clses[i]]
                color = COLORS.get(class_name, DEFAULT_COLOR)
                
                label = f"{class_name} {confs[i]:.2f}"
                if ids is not None: label = f"ID:{ids[i]} " + label
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                # 简单画文字背景
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1-th-5), (x1+tw, y1), color, -1)
                cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def _draw_statistics_panel(self, img):
        """实时 HUD 面板"""
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (220, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.putText(img, "Real-time Traffic", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y = 65
        for cls in sorted(self.vehicle_counts.keys()):
            count = len(self.vehicle_counts[cls])
            cv2.putText(img, f"{cls}: {count}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cls], 1)
            y += 20

    def _generate_report(self, report_path):
        """核心：二次数据挖掘可视化报告生成"""
        print(f"\n📊 正在生成数据挖掘报告: {report_path}")
        
        # 准备数据
        cls_data = {cls: len(ids) for cls, ids in self.vehicle_counts.items()}
        df_time = pd.DataFrame(self.time_series_data)

        # 创建画布 (包含两个子图)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决中文乱码
        
        # --- 图 1：车型分布饼图 ---
        if cls_data:
            labels = list(cls_data.keys())
            sizes = list(cls_data.values())
            # 将 OpenCV BGR 转换为 Matplotlib RGB
            pie_colors = [tuple(reversed([c/255 for c in COLORS[l]])) for l in labels]
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=pie_colors, shadow=True)
            ax1.set_title("各车型识别分布比例", fontsize=14)

        # --- 图 2：车流量随时间变化折线图 ---
        if not df_time.empty:
            ax2.plot(df_time['time'], df_time['count'], marker='o', linestyle='-', color='b', linewidth=2)
            ax2.fill_between(df_time['time'], df_time['count'], color='skyblue', alpha=0.3)
            ax2.set_xlabel("时间 (s)", fontsize=12)
            ax2.set_ylabel("累计检测数量 (台)", fontsize=12)
            ax2.set_title("车流量随时间增长趋势", fontsize=14)
            ax2.grid(True, linestyle='--', alpha=0.7)

        plt.suptitle(f"BIT_CLS 数据集系统推理分析报告\n(Total Vehicles: {sum(cls_data.values())})", fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图表
        plt.savefig(report_path)
        print(f"✨ 报告已保存至: {report_path}")

# ================= 🚀 主程序 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="输入路径")
    parser.add_argument('--model', default='./task1/best.pt')
    parser.add_argument('--out_dir', default='runs')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    filename = os.path.basename(args.input).split('.')[0]
    
    vid_out = os.path.join(args.out_dir, f"result_{filename}.mp4")
    report_out = os.path.join(args.out_dir, f"report_{filename}.png")

    engine = InferenceEngine(model_path=args.model)
    engine.run(args.input, vid_out, report_out)
import os
import time
import torch
import pandas as pd
from ultralytics import YOLO
import shutil

# ================= 🔧 配置区域 =================
# 1. 微调后的 YOLO 模型路径
MODEL_PATH = '/data2/zhuangyn/Deep-Learning-coding-task/task1/code/best.pt'
# 2. 你的 data.yaml 路径 (确保里面 test: 路径指向了正确的测试集)
DATA_YAML_PATH = '/data2/zhuangyn/Deep-Learning-coding-task/task1/dataset/BIT_YOLO_Dataset/data.yaml'
# 3. 结果输出目录
OUTPUT_ROOT = "runs/scheme_a_yolo_benchmark"
# ===============================================

def run_eval():
    # 准备目录
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT)

    # 1. 加载模型
    print(f"🔥 正在加载微调模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 2. 执行评测 (Validation mode on Test split)
    # split='test' 会让模型去读 data.yaml 中 test 路径下的数据
    print(f"🧪 正在测试集上执行全量评估...")
    
    # 计时开始
    t_start = time.time()
    
    # model.val 会自动计算 mAP, Precision, Recall 等
    results = model.val(
        data=DATA_YAML_PATH,
        split='test',      # 指定使用 test 集
        imgsz=640,         # 保持与训练一致
        conf=0.25,         # 置信度阈值
        iou=0.6,           # NMS IoU 阈值
        device=0,          # 指定 GPU ID
        save_json=True,    # 保存结果 json
        project=OUTPUT_ROOT,
        name='test_results'
    )
    
    t_end = time.time()

    # 3. 提取核心指标
    # metrics 包含多种精度数据
    metrics_dict = {
        "Model": "YOLO11_Scheme_A",
        "mAP50": results.box.map50,           # mAP at IoU=0.5
        "mAP50-95": results.box.map,         # mAP at IoU=0.5:0.95
        "Precision": results.box.mp,          # Mean Precision
        "Recall": results.box.mr,             # Mean Recall
        "Fitness": results.fitness            # 综合评价指标
    }

    # 4. 速度测试 (Inference Speed)
    # 利用 val 内部记录的时间
    speed_info = results.speed # 字典格式 {'preprocess': ms, 'inference': ms, 'loss': ms, 'postprocess': ms}
    total_latency_ms = speed_info['preprocess'] + speed_info['inference'] + speed_info['postprocess']
    fps = 1000 / total_latency_ms

    metrics_dict["Latency_ms"] = total_latency_ms
    metrics_dict["FPS"] = fps

    # 5. 保存并打印报表
    df = pd.DataFrame([metrics_dict])
    print("\n" + "="*60)
    print("🏆 方案 A (端到端 YOLO) 测试集评测报告")
    print("="*60)
    print(df.round(4).to_string(index=False))
    
    report_csv = os.path.join(OUTPUT_ROOT, "scheme_a_summary.csv")
    df.to_csv(report_csv, index=False)
    
    print(f"\n✅ 详细指标已保存至: {report_csv}")
    print(f"🖼️  检测可视化图片已保存至: {OUTPUT_ROOT}/test_results/")

    # 6. 单独输出每一类的指标 (方便分析 Sedan 效果)
    print("\n📊 逐类别详细指标:")
    class_names = model.names
    # results.box.p/r/ap 分别是每一类的 P, R, AP
    class_data = []
    for i, name in class_names.items():
        class_data.append({
            "Class": name,
            "Precision": results.box.p[i],
            "Recall": results.box.r[i],
            "AP50": results.box.ap50[i]
        })
    df_class = pd.DataFrame(class_data)
    print(df_class.round(4).to_string(index=False))
    df_class.to_csv(os.path.join(OUTPUT_ROOT, "scheme_a_class_metrics.csv"), index=False)

if __name__ == "__main__":
    run_eval()
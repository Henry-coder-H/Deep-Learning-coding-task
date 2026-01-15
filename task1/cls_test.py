import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import cv2
import shutil
from tqdm import tqdm
from PIL import Image

# ================= 🔧 硬件与路径配置 =================
# 必须指向 BIT_CLS_Dataset 下的 test 文件夹
TEST_DIR = '/data2/zhuangyn/Deep-Learning-coding-task/task1/dataset/BIT_CLS_Dataset/test' 
# 权重文件所在目录
WEIGHT_DIR = '/data2/zhuangyn/Deep-Learning-coding-task/task1/code/'
# 结果输出根目录
OUTPUT_ROOT = "runs/pure_cls_benchmark3"

# ⚠️ 严格锁定类别顺序，必须与你的 data.yaml 顺序完全一致
CLASS_NAMES = ['Bus', 'Microbus', 'Minivan', 'Sedan', 'SUV', 'Truck']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 强制映射关系：文件夹名称 -> 索引 (解决字母排序导致的 Sedan/SUV 易位问题)
TARGET_CLASS_TO_IDX = {
    'Bus': 0,
    'Microbus': 1,
    'Minivan': 2,
    'Sedan': 3,
    'SUV': 4,
    'Truck': 5
}
# ===================================================

# --- 1. 严格一致的预处理 (SquarePad) ---
class SquarePad:
    def __init__(self, target_size=224):
        self.target_size = target_size
    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        new_img = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        new_img.paste(img, ((self.target_size - new_w) // 2, (self.target_size - new_h) // 2))
        return new_img

# --- 2. 模型加载工厂 ---
def get_model(model_name, weight_path):
    print(f"📦 正在构建结构: {model_name} | 加载权重: {os.path.basename(weight_path)}")
    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))
    elif model_name == 'swin_t':
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, len(CLASS_NAMES))
    elif model_name == 'convnext_t':
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASS_NAMES))
    else:
        raise ValueError(f"未知模型: {model_name}")

    state_dict = torch.load(weight_path, map_location=DEVICE)
    clean_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state)
    return model.to(DEVICE).eval()

# --- 3. 测速函数 ---
def measure_speed(model):
    dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        for _ in range(30): _ = model(dummy_input) # 预热
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t_start = time.time()
    with torch.no_grad():
        for _ in range(200): _ = model(dummy_input)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    
    avg_latency = ((time.time() - t_start) / 200) * 1000
    fps = 1000 / avg_latency
    return fps, avg_latency

# --- 4. 可视化保存 ---
def save_result(img_tensor, p_idx, t_idx, conf, save_path):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    color = (0, 255, 0) if p_idx == t_idx else (0, 0, 255)
    cv2.rectangle(img, (0,0), (224, 40), (0,0,0), -1)
    txt = f"P:{CLASS_NAMES[p_idx]} G:{CLASS_NAMES[t_idx]} C:{conf:.2f}"
    cv2.putText(img, txt, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(save_path, img)

# --- 5. 执行单模型评测 ---
def evaluate_one_model(model_name, weight_file):
    weight_path = os.path.join(WEIGHT_DIR, weight_file)
    if not os.path.exists(weight_path):
        print(f"⚠️ 跳过 {model_name}: 找不到权重 {weight_path}")
        return None

    model_output_dir = os.path.join(OUTPUT_ROOT, model_name)
    vis_dir = os.path.join(model_output_dir, "vis_test_results")
    if os.path.exists(model_output_dir): shutil.rmtree(model_output_dir)
    os.makedirs(vis_dir)

    model = get_model(model_name, weight_path)
    fps, latency = measure_speed(model)

    # 数据预处理
    test_transform = transforms.Compose([
        SquarePad(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 【核心修复】：手动指定文件夹排序，禁止 ImageFolder 乱排
    # 这样可以确保标签 3 一定是 Sedan，标签 4 一定是 SUV
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)
    test_dataset.class_to_idx = TARGET_CLASS_TO_IDX
    # 重新生成样本列表以应用新的 class_to_idx
    test_dataset.samples = test_dataset.make_dataset(TEST_DIR, TARGET_CLASS_TO_IDX, extensions=('.jpg', '.jpeg', '.png'))
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    y_true, y_pred = [], []
    
    print(f"⌛ 正在推理... (共 {len(test_dataset)} 张图片)")
    for idx, (inputs, labels) in enumerate(tqdm(test_loader)):
        inputs = inputs.to(DEVICE)
        target = labels.item() 
        
        with torch.no_grad():
            outputs = model(inputs)
            prob = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(prob, 1)
        
        p = pred.item()
        y_true.append(target)
        y_pred.append(p)

        # 可视化前 200 张和所有错误的图片
        if idx < 200 or p != target:
            img_name = f"{idx:04d}_GT-{CLASS_NAMES[target]}_P-{CLASS_NAMES[p]}.jpg"
            save_result(inputs[0], p, target, conf.item(), os.path.join(vis_dir, img_name))

    # 指标计算
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    
    # 混淆矩阵图
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'{model_name} Test Set Result\nAcc: {acc:.4f} | Kappa: {kappa:.4f}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(os.path.join(model_output_dir, "confusion_matrix.png"))
    plt.close()

    # 保存报告
    pd.DataFrame(report).transpose().to_csv(os.path.join(model_output_dir, "test_report.csv"))

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Kappa": kappa,
        "Macro-F1": report['macro avg']['f1-score'],
        "Sedan-F1": report['Sedan']['f1-score'],
        "FPS": fps,
        "Latency(ms)": latency
    }

if __name__ == "__main__":
    eval_tasks = [
        {"name": "mobilenet", "file": "best_mobilenet.pth"},
        {"name": "resnet50",  "file": "best_resnet50.pth"},
        {"name": "swin_t",    "file": "best_swin_t.pth"},
        {"name": "convnext_t", "file": "best_convnext_t.pth"}
    ]

    summary = []
    for task in eval_tasks:
        res = evaluate_one_model(task["name"], task["file"])
        if res: summary.append(res)

    df = pd.DataFrame(summary)
    print("\n" + "="*80)
    print("🏆 BIT_CLS_Dataset [TEST SET] 最终评测报告 (已修复标签对齐)")
    print("="*80)
    print(df.round(4).to_string(index=False))
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    df.to_csv(os.path.join(OUTPUT_ROOT, "summary_report.csv"), index=False)
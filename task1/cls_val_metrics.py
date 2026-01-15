import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

# ================= 🔧 核心配置 =================
DATA_DIR = '/data2/zhuangyn/Deep-Learning-coding-task/task1/dataset/BIT_CLS_Dataset'
BATCH_SIZE = 32

# 1. 严格映射关系 (必须与训练一致)
TARGET_CLASS_TO_IDX = {
    'Bus': 0,
    'Microbus': 1,
    'Minivan': 2,
    'Sedan': 3,
    'SUV': 4,
    'Truck': 5
}
# 2. 类别名称列表 (用于绘图和报告)
CLASS_NAMES = list(TARGET_CLASS_TO_IDX.keys())
# ===============================================

class SquarePad:
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, img):
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = F.resize(img, (new_h, new_w), interpolation=F.InterpolationMode.BILINEAR)
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return F.pad(img, padding, fill=0)

def get_model(model_name, num_classes):
    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'swin_t':
        model = models.swin_t(weights=None)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif model_name == 'convnext_t':
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

def evaluate(model_type):
    model_path = f'best_{model_type}.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在准备评估: {model_type}")

    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到权重文件 {model_path}")
        return

    data_transforms = transforms.Compose([
        SquarePad(224), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # --- 1. 数据加载与强制对齐 ---
    val_path = os.path.join(DATA_DIR, 'val')
    val_dataset = datasets.ImageFolder(val_path, data_transforms)
    
    # 物理覆盖标签索引
    val_dataset.class_to_idx = TARGET_CLASS_TO_IDX
    val_dataset.samples = val_dataset.make_dataset(
        val_path, # 这里修正了变量名，确保能找到路径
        TARGET_CLASS_TO_IDX, 
        extensions=('.jpg', '.jpeg', '.png')
    )
    
    print(f"✅ 标签已强制对齐: {val_dataset.class_to_idx}")
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    num_classes = len(CLASS_NAMES)
    
    # 🔥🔥🔥 为了兼容你后续的可视化代码，这里定义一下小写的 class_names 🔥🔥🔥
    class_names = CLASS_NAMES 

    # --- 2. 模型加载 ---
    model = get_model(model_type, num_classes)
    try:
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print("✅ 权重加载成功！")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return
    
    model.to(device)
    model.eval()

    # 5. 推理并收集所有结果
    all_preds = []
    all_labels = []

    print("⏳ 正在进行全量推理...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 6. 计算高级指标
    # (1) 整体 Accuracy
    acc = accuracy_score(all_labels, all_preds)
    
    # (2) Cohen's Kappa
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    # (3) 详细报告 (这里开始使用 class_names，上面已经定义好了)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    print("\n" + "="*30)
    print(f"📊 评估结果报告 ({model_type})")
    print("="*30)
    print(f"Overall Accuracy:  {acc:.4f}")
    print(f"Cohen's Kappa:     {kappa:.4f}")
    print("-" * 30)

    # 7. 生成类似论文的表格 (DataFrame)
    data = []
    for cls in class_names:
        row = {
            'Class': cls,
            'Precision (查准率)': report_dict[cls]['precision'],
            'Recall (查全率)': report_dict[cls]['recall'],
            'F1-Score': report_dict[cls]['f1-score'],
            'Support (样本数)': report_dict[cls]['support']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    # 计算均值行
    mean_row = {
        'Class': 'Macro Avg',
        'Precision (查准率)': report_dict['macro avg']['precision'],
        'Recall (查全率)': report_dict['macro avg']['recall'],
        'F1-Score': report_dict['macro avg']['f1-score'],
        'Support (样本数)': '-'
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    print(df.round(4).to_string(index=False))
    
    # 保存表格到 CSV
    csv_filename = f'evaluation_metrics_{model_type}.csv'
    df.round(4).to_csv(csv_filename, index=False)
    print(f"\n✅ 详细指标已保存至 {csv_filename}")

    # 8. 画混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_type} Confusion Matrix (Acc: {acc:.2%}, Kappa: {kappa:.3f})')
    
    img_filename = f'confusion_matrix_{model_type}.png'
    plt.savefig(img_filename)
    print(f"✅ 混淆矩阵图已保存至 {img_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model', type=str, default='mobilenet', 
                        choices=['resnet50', 'mobilenet', 'swin_t', 'convnext_t'])
    args = parser.parse_args()
    
    evaluate(args.model)
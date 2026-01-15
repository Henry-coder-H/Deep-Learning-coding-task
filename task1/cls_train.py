"""
==============================================================================
文件名: cls_train.py
功能: 车型识别模型训练脚本 (Training Engine)

核心特性 (Key Features):
1. 几何特征保持: 实现了 SquarePad (Letterbox Resize) 类，通过填充黑边保持车辆长宽比，防止图像形变。
2. 类别不平衡处理: 针对 BIT-Vehicle 数据分布，采用 Weighted CrossEntropy Loss (加权损失)，提升少样本类别召回率。
3. 全全指标监控: 训练循环中实时计算 Loss, Accuracy, Precision, Recall, F1-Score, Cohen's Kappa。
4. 智能训练策略: 集成 Early Stopping (早停机制) 与 Learning Rate Decay (学习率衰减)。
5. 自动可视化: 训练结束后自动绘制并保存包含6个维度的训练曲线图 (training_curve_*.png)。

输入: 
    - BIT_CLS_Dataset (分类数据集)
    - MobileNetV3-Small  轻量级分类模型
    - Resnet50
输出: 
    - best_mobilenet.pth (基于 Kappa 指标的最优模型权重)
    - training_curve_mobilenet.png (训练过程指标变化图)
    - best_resnet50.pth (基于 Kappa 指标的最优模型权重)
    - training_curve_resnet50.png (训练过程指标变化图)
命令：
    - 训练 MobileNet 
    python cls_train.py --model mobilenet

    - 训练 ResNet-50 
    python cls_train.py --model resnet50

    - 训练 Swin
    python cls_train.py --model swin_t

    - 训练 Convnext
    python cls_train.py --model convnext_t

!!! 注意点：
    # 针对 Swin Transformer 的微调策略
    # lr=0.001 对 Swin 来说太大了，会导致不收敛
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    # 学习率衰减可以稍微温和一点
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
==============================================================================
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as F
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

# ================= 配置 =================
DATA_DIR = '/data2/zhuangyn/Deep-Learning-coding-task/task1/dataset/BIT_CLS_Dataset'
NUM_CLASSES = 6
BATCH_SIZE = 32
MAX_EPOCHS = 100       # 设置大一点，靠早停机制来控制
PATIENCE = 10          # 早停耐心值：10轮不提升就停
# =======================================

# 1. 自定义 Padding Resize (Letterbox) ---
class SquarePad:
    """
    将图片等比例缩放，长边对齐 target_size，短边填充黑色(0)以保持正方形。
    这是避免车辆变形的关键步骤！
    """
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, img):
        # 获取原始尺寸
        w, h = img.size
        # 计算缩放比例
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        # 等比例缩放
        img = F.resize(img, (new_h, new_w), interpolation=F.InterpolationMode.BILINEAR)
        # 计算填充量
        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        # Padding: (left, top, right, bottom)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        # 填充黑色
        return F.pad(img, padding, fill=0)

def get_model(model_name, num_classes):
    print(f"Loading model: {model_name}...")
    
    # === 1. ResNet50 ===
    if model_name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    # === 2. MobileNetV3 ===
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_small(weights='DEFAULT')
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    # === 3. Swin Transformer  ===
    elif model_name == 'swin_t':
        model = models.swin_t(weights='DEFAULT') 
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)

    # === 4. ConvNeXt Tiny  ===
    elif model_name == 'convnext_t':
        model = models.convnext_tiny(weights='DEFAULT')
        # ConvNeXt 的分类层通常是 classifier[2]
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

def plot_full_history(history, model_name):
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 创建一个 2行3列 的大图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{model_name} Training Metrics Analysis', fontsize=16)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend(); axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], label='Train')
    axes[0, 1].plot(epochs, history['val_acc'], label='Val')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend(); axes[0, 1].grid(True)

    # Cohen's Kappa
    axes[0, 2].plot(epochs, history['val_kappa'], color='purple', label='Val Kappa')
    axes[0, 2].set_title("Cohen's Kappa")
    axes[0, 2].legend(); axes[0, 2].grid(True)

    # Precision
    axes[1, 0].plot(epochs, history['val_precision'], color='orange', label='Val Precision (Macro)')
    axes[1, 0].set_title('Precision')
    axes[1, 0].legend();axes[1, 0].grid(True)
 
    # Recall
    axes[1, 1].plot(epochs, history['val_recall'], color='green', label='Val Recall (Macro)')
    axes[1, 1].set_title('Recall')
    axes[1, 1].legend();axes[1, 1].grid(True)

    # F1-Score
    axes[1, 2].plot(epochs, history['val_f1'], color='red', label='Val F1-Score (Macro)')
    axes[1, 2].set_title('F1-Score')
    axes[1, 2].legend();axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(f'training_curve_{model_name}.png')
    print(f" 训练全指标曲线已保存为 training_curve_{model_name}.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mobilenet', choices=['resnet50', 'mobilenet', 'swin_t', 'convnext_t'])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据增强 
    data_transforms = {
        'train': transforms.Compose([
            SquarePad(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            SquarePad(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 手动定义类别到索引的映射 
    # 必须保证 Sedan=3, SUV=4，从而对齐你的权重列表和 YOLO
    target_class_to_idx = {
        'Bus': 0,
        'Microbus': 1,
        'Minivan': 2,
        'Sedan': 3,
        'SUV': 4,
        'Truck': 5
    }

    image_datasets = {}

    for x in ['train', 'val']:
        ds = datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        ds.class_to_idx = target_class_to_idx
 
        ds.samples = ds.make_dataset(os.path.join(DATA_DIR, x), ds.class_to_idx, extensions=('.jpg', '.jpeg', '.png'))        # 必须加这一行，重新对齐 samples 里的 label
        image_datasets[x] = ds

    print("--- 检查类别对齐情况 (修正后) ---")
    print(image_datasets['train'].class_to_idx)
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    model = get_model(args.model, NUM_CLASSES)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    #  权重列表 
    cls_num_list = [558, 883, 476, 5922, 1392, 822]
    max_num = max(cls_num_list)
    weights = [max_num / x for x in cls_num_list]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)     # 学习率衰减：每 10 轮降低一半

    # 针对 Swin Transformer 的微调策略
    # lr=0.001 对 Swin 来说太大了，会导致不收敛
    # optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    # 学习率衰减可以稍微温和一点
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    # 历史记录器 
    history = {
        'train_loss': [], 'train_acc': [], 
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_kappa': []
    }
    
    best_kappa = 0.0 # 使用 Kappa 作为最佳模型的判断标准（比Acc更适合不平衡数据）
    epochs_no_improve = 0

    print(f"开始训练 (Model: {args.model}, Max Epochs: {MAX_EPOCHS})...")
    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        print(f'\nEpoch {epoch+1}/{MAX_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            # 收集用于计算高级指标的列表
            epoch_preds = []
            epoch_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'val':
                    epoch_preds.extend(preds.cpu().numpy())
                    epoch_labels.extend(labels.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # 记录与打印 
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            else:
                # 计算 Macro Average 指标
                p, r, f1, _ = precision_recall_fscore_support(epoch_labels, epoch_preds, average='macro', zero_division=0)
                kappa = cohen_kappa_score(epoch_labels, epoch_preds)
                
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                history['val_precision'].append(p)
                history['val_recall'].append(r)
                history['val_f1'].append(f1)
                history['val_kappa'].append(kappa)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Kappa: {kappa:.4f} F1: {f1:.4f}')

                #  保存最佳模型 
                if kappa > best_kappa:
                    best_kappa = kappa
                    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    torch.save(state_dict, f'best_{args.model}.pth')
                    epochs_no_improve = 0
                    print(f"新的最佳模型! (Kappa: {best_kappa:.4f})")
                else:
                    epochs_no_improve += 1
                    print(f"早停计数: {epochs_no_improve}/{PATIENCE}")

        if epochs_no_improve >= PATIENCE:
            print(f"\n 触发早停！")
            break

    time_elapsed = time.time() - start_time
    print(f'\n 训练结束! 总耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'最佳 Kappa: {best_kappa:.4f}')
    
    # 画图
    plot_full_history(history, args.model)

if __name__ == '__main__':
    main()
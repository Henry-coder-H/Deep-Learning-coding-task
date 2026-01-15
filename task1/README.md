# 车型识别任务：BIT-Vehicle 数据集处理与模型训练

本项目基于 **BIT-Vehicle 数据集** 实现了车辆检测与车型识别的完整流程。项目包含两种技术方案：**方案A（端到端目标检测）** 和 **方案B（级联式检测+分类）**。本 README 详细说明了数据集准备、代码结构与使用方式。

---

## 📁 数据集准备

### 1. 下载原始数据集
- **数据集名称**：BIT-Vehicle Dataset
- **内容**：包含 6 类车型（Bus, Microbus, Minivan, Sedan, SUV, Truck）
- **格式**：`.mat` 格式标注文件 + `.jpg` 图像
- **下载链接**：[BIT-Vehicle 官网](http://iitlab.bit.edu.cn/mcislab/vehicledb/) 或相关公开数据集平台

### 2. 数据集处理脚本

#### （1）`mat_to_yolo.py`  
将原始 `.mat` 标注转换为 YOLO 格式，生成 **BIT_YOLO_Dataset**，用于方案A训练。

**功能**：
- 读取 `VehicleInfo.mat`
- 将边界框转换为 YOLO 格式（归一化坐标）
- 按 7:2:1 划分训练集、验证集、测试集
- 生成 `data.yaml` 配置文件

**输出目录结构**：
```
BIT_YOLO_Dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

#### （2）`yolo_to_cls.py`  
将 YOLO 格式数据集裁剪为单车辆图像，生成 **BIT_CLS_Dataset**，用于方案B训练分类模型。

**功能**：
- 解析 YOLO 标签文件，裁剪每辆车
- 按类别归档至对应文件夹
- 保持车辆长宽比（使用 SquarePad 填充）

**输出目录结构**：
```
BIT_CLS_Dataset/
├── train/
│   ├── Bus/
│   ├── Microbus/
│   ├── Minivan/
│   ├── Sedan/
│   ├── SUV/
│   └── Truck/
├── val/
│   └── ...（同上）
└── test/
    └── ...（同上）
```

---

## 🧠 模型训练与评估

### 方案A：端到端目标检测（基于 YOLO）

#### `yolo_train.py`
使用 YOLOv11-large 进行端到端车型检测训练。

**核心训练策略**：
- 输入分辨率：1088×1088
- 数据增强：Mosaic, Mixup, HSV 增强，±10° 旋转
- 优化器：SGD，初始学习率 0.01
- 损失函数：CIoU
- 早停机制：Patience=10

**运行命令**：
```bash
python yolo_train.py
```

#### `yolo_test.py`
在测试集上评估训练好的 YOLO 模型。

**输出指标**：
- mAP@0.5、mAP@0.5:0.95
- Precision、Recall、FPS
- 逐类别 AP 与混淆矩阵

**运行命令**：
```bash
python yolo_test.py
```

---

### 方案B：级联式检测+分类

#### `cls_train.py`
训练分类模型，支持 4 种骨干网络：
- MobileNetV3-Small
- ResNet50
- Swin-Transformer-Tiny
- ConvNeXt-Tiny

**训练设置**：
- 输入：224×224（SquarePad 保持长宽比）
- 数据增强：水平翻转、旋转、颜色抖动
- 损失函数：加权交叉熵（应对类别不平衡）
- 优化器：Adam（CNN） / AdamW（Swin）
- 早停：Patience=10，监控 Kappa 系数

**运行命令**：
```bash
python cls_train.py --model resnet50   # 可选 mobilenet, swin_t, convnext_t
```

#### `cls_val_metrics.py`
在验证集上评估分类模型，输出详细指标与混淆矩阵。

**输出**：
- 准确率、Kappa、宏平均 Precision/Recall/F1
- 分类报告表格（CSV）
- 混淆矩阵图

**运行命令**：
```bash
python cls_val_metrics.py --model resnet50
```

#### `cls_test.py`
在测试集上全面评估分类模型，包含推理速度测试。

**输出**：
- 准确率、Kappa、F1-Score
- 逐类别召回率与精确率
- 推理延迟（ms）与 FPS
- 可视化错误样本

**运行命令**：
```bash
python cls_test.py
```

---

## 📊 实验结果摘要（来自报告）

### 方案A（YOLO11-large）
- **mAP@0.5**：97.73%
- **mAP@0.5:0.95**：96.6%
- **FPS**：159.9（RTX 5070）
- **优势**：端到端、高速度、适合实时场景

### 方案B（ResNet50 分类器）
- **准确率**：97.80%
- **Kappa**：0.967
- **Macro-F1**：0.963
- **优势**：对相似车型（Sedan/SUV）判别力更强，适合离线高精度场景

---

## 🛠 环境依赖

- Python 3.8+
- PyTorch 2.x
- Ultralytics (YOLOv11)
- OpenCV, scipy, sklearn, tqdm
- 显卡：NVIDIA GPU（建议 ≥ 8GB 显存）

安装依赖：
```bash
pip install torch torchvision ultralytics opencv-python scipy scikit-learn tqdm pandas seaborn matplotlib
```

---

## 📌 注意事项

1. 数据集路径需在代码中根据实际情况修改。
2. 训练分类模型时，务必保持 `CLASS_NAMES` 顺序与 `data.yaml` 一致。
3. Swin-Transformer 需使用较低学习率（5e-5），否则易发散。
4. 测试集评估前请确保模型权重路径正确。

---

## 📄 文件说明

| 文件 | 功能 |
|------|------|
| `mat_to_yolo.py` | 原始数据集转 YOLO 格式 |
| `yolo_to_cls.py` | YOLO 数据集裁剪为分类数据集 |
| `yolo_train.py` | YOLO 检测模型训练 |
| `yolo_test.py` | YOLO 模型测试评估 |
| `cls_train.py` | 分类模型训练 |
| `cls_val_metrics.py` | 分类模型验证集评估 |
| `cls_test.py` | 分类模型测试集评估 |

---

## 📚 参考文献

- BIT-Vehicle Dataset: [论文链接](http://iitlab.bit.edu.cn/mcislab/vehicledb/)
- YOLOv11: [Ultralytics 文档](https://docs.ultralytics.com/)
- ResNet, Swin, ConvNeXt: PyTorch 官方实现

---

如遇问题，请检查路径设置、依赖版本及数据集完整性。建议按顺序执行脚本，确保数据流正确传递。
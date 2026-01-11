# 车牌识别系统 (YOLOv8-Pose + ResNet-CRNN)

## 1. 项目概述
本项目实现了一个高精度、鲁棒的车牌识别（LPR）系统，专为复杂场景设计。
- **检测（Detection）**：使用 **YOLOv8-Pose** 检测车牌的 4 个角点（关键点），支持倾斜车牌定位。
- **矫正（Rectification）**：根据检测到的 4 个角点进行**透视变换**，将倾斜的车牌“拉正”。
- **识别（Recognition）**：使用工业级改进版 **ResNet18 + BiLSTM + CTC (CRNN)** 对矫正后的车牌图片进行文字识别。

**主要特性：**
*   **高精度**：测试集全字匹配率达 **98.7%**。
*   **高鲁棒性**：在困难场景（模糊、倾斜、阴影等）下全字匹配率仍达 **96.4%**。
*   **实时性**：端到端推理速度 > 70 FPS。

## 2. 目录结构说明
```
task2_member1/
├── configs/            # YOLO 训练配置
│   └── data_ccpd_kpts.yaml
├── data/               # 数据存放目录 (需自行生成)
├── detection/          # 检测模块
│   ├── inference.py    # YOLO 推理封装类
│   └── train_yolo_pose.py # YOLO 训练脚本
├── recognition/        # 识别模块
│   ├── model.py        # ResNet-CRNN 模型定义
│   ├── train_crnn.py   # CRNN 训练脚本
│   ├── datasets.py     # 数据集加载
│   └── inference.py    # CRNN 推理封装类
├── pipeline/           # 端到端流水线
│   ├── infer_pipeline.py # 视频/图片推理脚本
│   └── utils.py        # 通用工具 (透视变换等)
├── evaluation/         # 评估模块
│   └── evaluate_pipeline.py # 指标评估脚本
├── scripts/            # 数据预处理脚本
│   ├── prepare_detection_data.py # 生成 YOLO 数据
│   └── prepare_recognition_data.py # 生成 CRNN 数据
├── runs/               # 训练日志与权重保存路径 (本地生成)
├── raw_ccpd/           # 原始数据集存放处 (不上传)
├── requirements.txt    # 依赖包
└── README.md           # 本文档
```

## 3. 环境配置与安装

### 前置要求
- Python 3.10+
- CUDA 11.8+

### 安装步骤
1. **创建环境**：
   ```bash
   conda create -n lpr python=3.10 -y
   conda activate lpr
   ```

2. **安装依赖**：
   ```bash
   # 安装 PyTorch (根据你的 CUDA 版本调整)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # 安装项目依赖
   pip install -r requirements.txt
   ```

## 4. 模型权重下载

由于模型权重文件较大，未包含在 GitHub 仓库中。请从以下 Google Drive 链接下载，并按说明放置：

**下载链接**: [在此处插入 Google Drive 链接]

**放置位置**:
1.  下载 `best.pt` (YOLO 检测权重)，放入 `runs/pose/train/weights/` 目录（如果目录不存在请手动创建）。
2.  下载 `best.pth` (CRNN 识别权重)，放入 `runs/rec/` 目录。

```bash
mkdir -p runs/pose/train/weights
mkdir -p runs/rec
# 将下载的文件移动到上述目录
```

## 5. 数据集准备

本项目基于 **CCPD2019 / CCPD2020** 数据集。
请确保将数据集解压至 `raw_ccpd` 目录下，并准备好 `all_train.txt`, `all_test.txt`, `all_hardtest.txt` 图片列表文件。

**生成训练数据：**
```bash
# 1. 准备检测数据 (YOLO格式)
python scripts/prepare_detection_data.py \
  --ccpd-root raw_ccpd \
  --out-root data/det \
  --train-list all_train.txt \
  --test-list all_test.txt

# 2. 准备识别数据 (CRNN格式 - 裁剪出车牌)
python scripts/prepare_recognition_data.py \
  --ccpd-root raw_ccpd \
  --out-root data/rec \
  --train-list all_train.txt \
  --test-list all_test.txt \
  --imgw 160 --imgh 32
```

## 6. 模型训练

**训练参数说明：**
*   **Detection**: YOLOv8-Pose, Epochs=50, Batch=32, Imgsz=640
*   **Recognition**: ResNet18-CRNN, Epochs=20, Batch=256, Imgsz=160x32, Optimizer=Adam, LR=1e-3

**训练命令：**
```bash
# 训练 YOLOv8-Pose
CUDA_VISIBLE_DEVICES=0 python detection/train_yolo_pose.py \
  --data configs/data_ccpd_kpts.yaml \
  --epochs 50 --batch 32 --device 0

# 训练 CRNN
CUDA_VISIBLE_DEVICES=0 python recognition/train_crnn.py \
  --train-labels data/rec/train_labels.txt \
  --val-labels data/rec/val_labels.txt \
  --imgw 160 --imgh 32 --batch 256 --epochs 20 --device 0
```

## 7. 使用方法 (Inference)

支持图片和视频的端到端推理。

**单张图片/视频推理：**
```bash
python pipeline/infer_pipeline.py \
  --det-weights runs/pose/train/weights/best.pt \
  --rec-weights runs/rec/best.pth \
  --source test.jpg \
  --save-vis  # 保存可视化结果
```
*   输出结果将保存在 `runs/predict/` 目录下。
*   `--source` 可以是图片路径，也可以是视频路径 (`.mp4`)。

## 8. 性能评估 (Evaluation)

### 评估指标结果

| 数据集 | 全字匹配率 (Full Match Accuracy) | 字符准确率 (Char Accuracy) | FPS | 备注 |
| :--- | :---: | :---: | :---: | :--- |
| **All Test** (常规测试集) | **98.70%** | **99.71%** | **73.9** | 满足 Accuracy > 95%, FPS > 30 |
| **Hard Test** (困难/鲁棒性测试集) | **96.44%** | **99.38%** | **70.9** | 在复杂场景下保持极高鲁棒性 |

*(注：Hard Test 包含倾斜、模糊、天气变化等复杂场景，测试量 30400 张)*

**复现评估结果：**

1.  **评估常规测试集**：
    ```bash
    python evaluation/evaluate_pipeline.py \
      --det-weights runs/pose/train/weights/best.pt \
      --rec-weights runs/rec/best.pth \
      --img-root raw_ccpd \
      --img-list all_test.txt
    ```

2.  **评估鲁棒性测试集**：
    ```bash
    python evaluation/evaluate_pipeline.py \
      --det-weights runs/pose/train/weights/best.pt \
      --rec-weights runs/rec/best.pth \
      --img-root raw_ccpd \
      --img-list all_hardtest.txt
    ```

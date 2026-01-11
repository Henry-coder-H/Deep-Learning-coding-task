# 车牌识别系统（ YOLOv8 + LPRNet ）

本项目为两阶段的轻量级车牌识别系统。系统使用 **YOLOv8** 进行车牌检测后使用改进的 **LPRNet** 进行端到端的字符识别。在 **CCPD (Chinese City Parking Dataset)** 数据集上进行了训练和测试，支持倾斜、模糊、光照变化等复杂场景。



## 技术路线与致谢

本项目基于优秀的开源工作进行了二次开发与适配：

1.  **车牌检测**
    
    * 模型：**YOLOv8**
    * 预训练权重来源：[Automatic-License-Plate-Recognition-using-YOLOv8](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8)

2.  **车牌识别**
    
    * 模型：**LPRNet**
    * 代码框架来源：[LPRNet_Pytorch by sirius-ai](https://github.com/sirius-ai/LPRNet_Pytorch)
    * 利用 CCPD 数据集的标注信息（Ground Truth）将车牌边界框区域作为模型输入，重新训练了 LPRNet 模型，使其更适应检测器输出的未校正仅框取图像。
    
3.  **数据集**
    * 训练与测试均基于 **CCPD 数据集**。
    * 数据集来源：[CCPD (Chinese City Parking Dataset)](https://github.com/detectRecog/CCPD)



## 项目结构

```
├── weights/                         # 模型权重
│   ├── license_plate_detector.pt    # YOLOv8 车牌检测权重 (来源见上文)
│   ├── lprnet_best.pth           	 # LPRNet 字符识别最佳权重 (本项目训练)
│   └── ...
│
├── LPRNet_Pytorch/                  # LPRNet 核心代码 (源自 sirius-ai)
│   ├── model/                       # 模型定义
│   ├── data/                        # 数据加载与预处理
│   ├── train_LPRNet.py              # LPRNet 训练脚本
│   └── ...
│
├── splits/             			 # 数据集分割文件
|   ├── all_train.txt
|   ├── all_test.txt
|   └── val.txt
├── val.py                           # 核心评估脚本 (检测+识别全流程)
├── test-yolo.py                     # 单独测试 YOLO 检测效果
├── test_lprnet_only.py              # 单独测试 LPRNet 识别效果
├── test-vedio.py                    # 视频推理脚本
├── test.py                   		 # 图片推理脚本
├── kaggle_train.py                  # Kaggle 平台训练脚本
└── README.md
```



## 快速开始

### 1. 环境配置

建议使用 Python 3.10+ 环境：

```bash
# 安装 PyTorch (请根据你的 CUDA 版本调整命令)
pip install torch torchvision torchaudio

# 安装其他依赖
pip install opencv-python ultralytics numpy imutils psutil
```

### 2. 数据准备

请下载 CCPD2019 和 CCPD2020 数据集，并按照以下结构放置在项目根目录：

```Plaintext
CCPD/
├── ccpd_base/
├── ccpd_blur/
├── ccpd_challenge/
├── ccpd_db/
├── ccpd_fn/
├── ccpd_green/         # CCPD2020 新能源车牌
├── ccpd_rotate/
├── ccpd_tilt/
├── ccpd_weather/
└── splits/             # 数据集分割文件
    ├── all_train.txt
    ├── all_test.txt
    └── val.txt
```

### 3. 运行测试

**单张图片推理：**

```bash
python test.py --source your_image.jpg
```

**视频流推理：**

```bash
python test-vedio.py --source your_video.mp4
```



## 性能指标

基于 `lprnet_best.pth` 权重在 CCPD 测试集上的表现：

|              数据集               |  数量  | 全字匹配率 (Full Match Accuracy) | 字符准确率 (Char Accuracy) |    FPS    |
| :-------------------------------: | :----: | :------------------------------: | :------------------------: | :-------: |
|     **All Test** (常规测试集)     | 135402 |            **85.13%**            |         **93.14%**         | **24.10** |
| **Hard Test** (困难/鲁棒性测试集) | 30400  |            **59.44**             |         **79.59%**         | **29.94** |

*(注：FPS 在不同硬件条件下会有显著区别)*

**复现评估结果：**

```bash
python val.py
```

**快速评估 (随机抽样 500 张)：** 修改 `val.py` 中 `RANDOM_SAMPLE = True`

# 智能交通识别系统 - UI 应用

基于 Streamlit 开发的单页 Web 应用，集成 **车型识别**、**车牌识别**、**车速识别** 三大核心功能。

## 📁 项目结构

```
UI_App/
├── app.py                      # Streamlit 主应用
├── requirements.txt            # Python 依赖
├── README.md                   # 本文档
├── models/                     # 模型模块
│   ├── __init__.py
│   ├── plate_recognizer.py     # 车牌识别 (YOLOv8-Pose + CRNN)
│   ├── speed_estimator.py      # 车速估计 (YOLOv11 + 单应性变换)
│   └── vehicle_classifier.py   # 车型分类 (占位实现)
└── weights/                    # 模型权重目录 (需手动创建并放入权重)
    ├── plate_detect.pt         # 车牌检测模型 (YOLOv8-Pose)
    ├── plate_rec.pth           # 车牌识别模型 (ResNet-CRNN)
    ├── yolov11l.pt             # 车辆检测模型 (YOLOv11l)
    └── vehicle_classifier.pth  # 车型分类模型 (可选)
```

## 🔧 环境配置

### 1. 安装依赖

```bash
cd UI_App
pip install -r requirements.txt
```

### 2. PyTorch 安装 (根据 CUDA 版本)

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio
```

## ⚖️ 模型权重配置

**请在 `UI_App/weights/` 目录下放置以下权重文件：**

### 必需的权重文件

| 文件名 | 说明 | 来源 |
|--------|------|------|
| `plate_detect.pt` | 车牌检测模型 (YOLOv8-Pose) | 来自 `task2_member1/runs/pose/train/weights/best.pt` |
| `plate_rec.pth` | 车牌识别模型 (ResNet-CRNN) | 来自 `task2_member1/runs/rec/best.pth` |
| `yolov11l.pt` | 车辆检测模型 (YOLOv11l) | 从 Ultralytics 下载 `yolo11l.pt` 并重命名 |

### 可选的权重文件

| 文件名 | 说明 | 备注 |
|--------|------|------|
| `vehicle_classifier.pth` | 车型分类模型 | 暂未实现，当前使用基于 COCO 类别的简单分类 |

### 权重放置步骤

```bash
# 1. 创建 weights 目录
mkdir -p UI_App/weights

# 2. 复制车牌识别权重
cp task2_member1/runs/pose/train/weights/best.pt UI_App/weights/plate_detect.pt
cp task2_member1/runs/rec/best.pth UI_App/weights/plate_rec.pth

# 3. 下载 YOLOv11l 权重 (从 Ultralytics 官网)
# 下载后重命名为 yolov11l.pt 放入 weights 目录
```

## 🚀 运行应用

```bash
cd UI_App
streamlit run app.py
```

应用将在浏览器中打开，默认地址: `http://localhost:8501`

## 📖 使用说明

### 功能概述

| 功能 | 图片支持 | 视频支持 | 说明 |
|------|:-------:|:-------:|------|
| 车型识别 | ✅ | ✅ | 识别车辆类型 (轿车/SUV/货车等) |
| 车牌识别 | ✅ | ✅ | 检测并识别车牌号码 |
| 车速识别 | ❌ | ✅ | 估算车辆行驶速度 (需标定) |

### 操作流程

#### 图片模式
1. 在侧边栏上传图片文件 (jpg/png/jpeg)
2. 勾选需要的功能 (车型识别/车牌识别)
3. 点击"开始检测"按钮
4. 查看处理结果和统计表格
5. 可下载处理后的图片

#### 视频模式
1. 在侧边栏上传视频文件 (mp4/avi/mov)
2. 勾选需要的功能
3. 如勾选"车速识别"，需要先完成坐标标定：
   - 系统显示视频第一帧
   - 输入至少4个标定点的像素坐标和真实世界坐标 (厘米)
   - 点击"确认标定"完成标定
4. 点击"开始检测"按钮
5. 等待进度条完成
6. 查看/下载处理后的视频

### 车速标定说明

车速识别需要进行相机标定，将像素坐标转换为真实世界坐标。

**标定参考数据：**
- 高速公路虚线：6m线段 + 9m间隔
- 城市快速路虚线：4m线段 + 4m或6m间隔
- 标准车道宽度：3.75m
- 人行横道最小宽度：3m

**标定步骤：**
1. 选择视频画面中的特征点 (车道线端点、斑马线角点等)
2. 记录像素坐标 (从图像中读取)
3. 估算真实世界坐标 (单位：厘米)
4. 建议至少标注6个点，覆盖画面的不同深度

## 🎨 界面说明

### 侧边栏 (控制区)
- 文件上传框
- 功能复选框 (车型/车牌/车速)
- 开始检测按钮
- 系统信息显示

### 主展示区 (结果区)
- 默认：欢迎语和功能介绍
- 处理中：进度条
- 完成后：
  - 图片模式：处理后的图片 + 统计表格 + 下载按钮
  - 视频模式：视频播放器 + 下载按钮

### 视觉标注规范
- 每辆车一个检测框
- 颜色按车辆ID分配 (同一辆车保持颜色一致)
- 标签显示在框顶部，格式：`[车型 | 车牌号 | 速度]`

## ⚠️ 注意事项

1. **首次运行**会下载模型，可能需要等待
2. **GPU 推荐**：有 CUDA 的环境推理速度更快
3. **视频处理**需要将整个视频处理完毕后才能显示
4. **车速精度**取决于标定质量，建议多标几个点并验证

## 🔍 技术实现

- **车牌识别**: YOLOv8-Pose (检测4角点) + 透视变换 + ResNet18-BiLSTM-CTC
- **车速识别**: YOLOv11l (车辆检测) + 简单IoU追踪 + 单应性变换 + 卡尔曼滤波
- **车型识别**: 基于 COCO 类别的简单映射 (完整模型待实现)

## 📝 更新日志

- **v1.0.0** (2026-01-14): 初始版本，集成车牌识别和车速识别

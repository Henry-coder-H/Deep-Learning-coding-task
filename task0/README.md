# Task 0: YOLO 模型性能评估

## 代码说明

本项目是深度学习作业的 Task 0 部分。该脚本 (`yolo_detection_mission_0.py`) 的主要目的是在**不进行微调训练**的情况下，直接使用官方预训练的 **YOLOv8** 和 **YOLOv11** 系列模型（从 nano 到 x-large）对车辆检测数据集进行 Zero-shot 性能评估。

### 核心功能

- **多模型对比**：自动下载并评估 10 个模型版本（YOLOv8n/s/m/l/x 和 YOLOv11n/s/m/l/x）。
- **智能过滤**：由于使用预训练模型，脚本会自动检测并**只评估 COCO 数据集包含的类别**（如 Car, Bus, Truck 等），忽略数据集中模型未见过的自定义类别。
- **可视化报告**：自动生成 mAP 对比图、推理速度对比图、热力图以及详细的 Excel/CSV 数据表。

## 环境依赖

请确保你的 Python 环境安装了以下依赖库：

```bash
pip install ultralytics torch torchvision pandas matplotlib seaborn openpyxl tqdm pyyaml
```

- **Python**: >= 3.8
- **CUDA**: 推荐使用 GPU 进行评估（脚本会自动检测 CUDA，如果不可用则使用 CPU）。

## 目录结构

在运行代码前，请确保文件目录结构如下：

Plaintext

```
Project_Root/
├── yolo_detection_mission_0.py    # 主评估脚本
├── VehiclesDetectionDataset/      # 数据集目录
│   └── dataset.yaml               # 数据集配置文件
└── results/                       # (运行后自动生成) 存放评估结果和图表
```

> **注意**：请确保 `dataset.yaml` 路径正确，且文件内的路径配置指向了真实的图片/标签位置。

## 运行代码

在终端或命令行中执行以下命令：

```bash
python yolo_detection_mission_0.py
```

脚本将按照以下流程运行：

1. 加载数据集配置。
2. 依次下载并加载 10 个 YOLO 预训练模型。
3. 对每个模型进行验证（Validation），计算 mAP、F1 Score 和推理耗时。
4. 在控制台输出进度，并将最终结果保存到本地。

## 输出结果

评估完成后，结果将保存在 `results/vehicle_detection_eval_filtered` 目录下：

1. **可视化图表 (`model_comparison.png`)**:
   - 包含 mAP50-95 精度对比。
   - 推理速度（毫秒/张）对比。
   - 不同尺寸模型的性能曲线。
2. **数据表格**:
   - `model_evaluation_results.xlsx`: 包含详细指标、过滤信息汇总。
   - `model_evaluation_results.csv`: 原始数据记录。
3. **文本报告 (`performance_report.txt`)**:
   - 自动生成的总结报告，推荐“最佳精度模型”、“最快速度模型”和“最佳平衡模型”。

## 注意事项

- **评估模式**: 本代码仅进行**验证 (Validation)**，不包含训练过程。
- **类别匹配**: 评估指标仅基于模型在 COCO 数据集中学到的 80 个类别。如果你的数据集中包含特殊类别（如 "emergency_vehicle"），且该类别不在 COCO 列表中，它将在评估时被忽略。
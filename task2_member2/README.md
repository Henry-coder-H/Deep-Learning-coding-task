# 车牌识别系统（PP-OCRv5）
本项目围绕**CCPD数据集**打造了一套适配复杂路况的实用型车牌识别系统，通过定向优化检测、矫正、识别全链路流程，实现了高精度与强鲁棒性的平衡。

核心技术链路：

- **车牌检测**：以PP-OCRv5文本检测模型为基础，叠加车牌专属的几何约束（如长宽比、区域占比）与候选框过滤规则，解决了不同角度、尺度下车牌的精准定位问题。
- **字符识别**：基于PP-OCRv5_rec模型，针对CCPD涵盖的“省份简称+字母+数字”车牌字符组合进行专项训练，让模型适配国内车牌的字符分布特性。

核心亮点：

- **识别精度突出**：在真值框输入的理想条件下，测试集整牌全对准确率达99.30%、字符平均准确率99.78%；端到端实际场景中，整牌识别准确率达90.04%。
- **复杂场景抗干扰强**：面对强透视、模糊等困难样本子集，识别模型的整牌匹配率仍保持98.21%，较通用模型（45.23%）提升超1倍。
- **推理效率达标**：单识别环节推理速度超150 FPS，端到端全流程达20.38 FPS，可支撑视频流的实时车牌识别。

## 目录说明

```
TASK2_MEMBER2
├── datasets/                 # 数据集目录（需自行解压配置CCPD数据集）
├── PP-OCRv5_server_rec/      # 训练完成的车牌识别模型权重目录（推理/评估必需）
│   ├── inference.json        
│   ├── inference.pdiparams  
│   └── inference.yml        
├── splits/                   # 数据集划分列表文件目录（指定训练/测试/难例样本）
│   ├── all_hardtest.txt      
│   ├── all_test.txt          
│   └── all_train.txt        
├── train/                    # 模型微调所需补充文件（适配PaddleOCR官方仓库）
│   ├── __init__.py           # 包初始化文件，用于导入自定义数据增强算子
│   ├── ccpd_warp.py          # 自定义CCPD车牌透视矫正算子（解决倾斜/变形问题）
│   ├── make_gt.py            # 数据预处理脚本（解析标注、生成label文件、构建字符字典）
│   └── PP-OCRv5_server_rec_ccpd.yml  # 适配CCPD数据集的模型训练配置文件（指定超参/数据路径）
├── evaluate_train_rec.py     # 仅识别模块评估脚本
├── evaluate_train.py         # 端到端全流程评估脚本
├── test.py                   # 图片/视频端到端车牌识别脚本
└── README.md             
```

## 第一部分：推理部署（快速使用，无需训练）

该部分面向仅需使用训练好的模型完成图片/视频车牌检测识别的场景，提供轻量化环境配置与部署方案，无需搭建复杂训练环境。

### 1. 推理环境配置
#### 1.1 基础环境要求
- **Python**：3.9 / 3.10 / 3.11 / 3.12 / 3.13
- **CUDA**：11.8 / 12.6 / 12.9 / 13.0

#### 1.2 轻量化依赖安装
直接执行以下命令，完成PaddlePaddle GPU版与核心推理依赖的安装：
```bash
# 步骤1：安装适配CUDA 11.8的PaddlePaddle（其他CUDA版本可替换对应镜像源）
python3 -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 步骤2：安装剩余推理必需依赖
pip install paddleocr opencv-python ultralytics numpy imutils psutil Pillow
```
### 2. 数据准备
- 数据集：CCPD2019 + CCPD2020 新增绿牌子集
- 存放要求：将数据集解压至 `datasets` 目录下，并将 `all_train.txt`, `all_test.txt`, `all_hardtest.txt` 图片列表文件存于 `splits` 目录下
- 目录结构：
  ```
  ├── datasets
  │   ├── ccpd_base
  │   ├── ccpd_green
  │   ├── ...
  │   └── splits
  │       ├── all_train.txt
  │       ├── all_test.txt
  │       ├── all_hardtest.txt
  │       └── ...
  ```
### 3. 使用说明
本系统支持**图片/视频格式文件**的端到端车牌识别测评，操作步骤如下：

#### 3.1 前置准备
运行前请确保 `PP-OCRv5_server_rec` 模型文件夹放置于 `test.py` 脚本的同级目录下。

#### 3.2 运行命令
##### 3.2.1 完整参数运行（指定输出结果）
执行以下命令，可自定义输出文件名与保存目录：
```bash
python test.py --source my_car.jpg --out detected_car --out_dir ./outputs
```

##### 3.2.2 默认参数运行（快速测试）
若不指定输出名称与保存路径，识别结果将以默认名称 `output` 保存至脚本同级目录：
```bash
python test.py --source my_car.jpg
```

#### 3.3 补充说明
运行脚本时，程序将自动检测并下载安装其他所需的依赖模型于同级目录 `.paddlex`文件夹下，无需手动配置。

### 4. 模型评估

注：须确保目录下已有将 `PP-OCRv5_server_rec` 模型文件夹与以下文件夹结构（test.py脚本运行时自动下载）

```
.paddlex
├── fun_cet
├── locks
└── official_models
    ├── PP-OCRv5_server_det
```

#### 4.1 端到端评估（`evaluate_train.py`）
流程：检测模型定位车牌 + 识别模型识别字符，模拟真实场景下的完整车牌识别流程。
```bash
python evaluate_train.py
```

核心指标（all_test 集）：
```
评测图片数量: 135402
- 整牌全对准确率：90.04%
- 字符平均准确率：94.05%
- FPS=20.38
- hardtest 整牌匹配率: 64.87%   (样本数=30400)
```

#### 4.2 仅识别评估（`evaluate_train_rec.py` ）
流程：使用 CCPD 真值框裁剪车牌，排除检测环节干扰，仅评估识别模型的性能上限。
```bash
python evaluate_train_rec.py
```

核心指标（all_test 集）：
| 模型版本   | 整牌全对准确率 | 字符平均准确率 | FPS    | hardtest 整牌匹配率 |
| ---------- | -------------- | -------------- | ------ | ------------------- |
| 初始模型   | 78.88%         | 88.44%         | 127.38 | 45.23%              |
| 训练后模型 | 99.30%         | 99.78%         | 149.22 | 98.21%              |


## 第二部分：模型训练（自定义微调）
该部分面向需要基于CCPD数据集微调PP-OCRv5_rec模型、优化模型效果的场景，包含完整的数据准备、训练配置、模型导出。

### 1. 训练环境配置
在完成上述**推理环境配置**的基础上，额外搭建训练所需的依赖与仓库环境。

#### 1.1 PaddleOCR 及训练依赖安装
```bash
# 克隆PaddleOCR仓库（训练必需，包含训练脚本、配置文件等）
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR

# 安装完整依赖（含训练、数据处理、评估组件，使用清华源加速）
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 1.2 预训练模型下载
```bash
wget -c https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams
```

### 2. 数据准备
#### 2.1 原始数据存放
- 数据集：CCPD2019 + CCPD2020 新增绿牌子集
- 存放要求：将数据集解压至 `datasets` 目录下，并将 `all_train.txt`, `all_test.txt`, `all_hardtest.txt` 图片列表文件存于 `splits` 目录下
- 目录结构：
  ```
  ├── datasets
  │   ├── ccpd_base
  │   ├── ccpd_green
  │   ├── ...
  │   └── splits
  │       ├── all_train.txt
  │       ├── all_test.txt
  │       ├── all_hardtest.txt
  │       └── ...
  ```

#### 2.2 训练数据预处理
> [!CAUTION]
> 以下代码相关配置路径在PaddleOCR官方仓库中进行，拉取PaddleOCR官方仓库后，须自行将本项目train文件夹下相关代码放置/替换在指定的目录下。

##### 2.2.1 新增脚本：`tools/make_gt.py`
```
python make_gt.py
```

功能：

1.  解析 CCPD 文件名中的车牌字符（省份简称 + 字母 + 数字）；
2.  读取划分文件，生成 PaddleOCR 识别训练用 label 文件：
    - `train_data/rec/ccpd/rec_gt_train.txt`
    - `train_data/rec/ccpd/rec_gt_val.txt`（从训练集随机取5%）
    - `train_data/rec/ccpd/rec_gt_test.txt`
3.  统计字符分布，生成车牌专用字典：`train_data/rec/ccpd/plate_dict.txt`

##### 2.2.2 划分训练/验证集
直接在终端运行以下命令，从训练集中抽取5%作为验证集：
```bash
shuf train_data/rec/ccpd/rec_gt_train_all.txt > /tmp/ccpd_train_shuf.txt
TOTAL=$(wc -l < /tmp/ccpd_train_shuf.txt)
VAL=$((TOTAL / 20))   # 5% 作为验证集
head -n $VAL /tmp/ccpd_train_shuf.txt > train_data/rec/ccpd/rec_gt_val.txt
tail -n +$((VAL + 1)) /tmp/ccpd_train_shuf.txt > train_data/rec/ccpd/rec_gt_train.txt
```

##### 2.2.3 新增脚本：`ppocr/data/imaug/ccpd_warp.py`
自定义 `CCPDWarp` 算子，根据文件名中的车牌四点坐标完成透视矫正，解决车牌倾斜、透视变形问题。

##### 2.2.4 修改脚本：`ppocr/data/imaug/__init__.py`
添加 `CCPDWarp` 算子导入，使配置文件可正常调用该自定义数据增强算子。

### 3. 模型训练与导出
#### 3.1 配置文件修改
将 `configs/rec/PP-OCRv5/PP-OCRv5_server_rec_ccpd.yml` 替换为本项目train目录下的同名配置文件（已适配CCPD数据集与车牌识别场景）。

#### 3.2 训练命令
```bash
python3 -u tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec_ccpd.yml \
  -o Global.pretrained_model=./pretrain/PP-OCRv5_server_rec_pretrained.pdparams \
     Global.save_model_dir=./output/ccpd_ppocrv5_server_rec \
     Global.epoch_num=10 \
     Train.loader.num_workers=0 \
     Eval.loader.num_workers=0 \
  2>&1 | tee train_$(date +%F_%H%M).log
```

#### 3.3 模型导出
训练完成后，导出推理可用的模型文件：
```bash
python3 tools/export_model.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec_ccpd.yml \
  -o Global.pretrained_model=./output/ccpd_ppocrv5_server_rec/best_accuracy \
     Global.save_inference_dir=./ccpd_ppocrv5_server_rec
```

导出后目录结构：
```text
./ccpd_ppocrv5_server_rec/
├── inference.json
├── inference.pdiparams
└── inference.yml
```


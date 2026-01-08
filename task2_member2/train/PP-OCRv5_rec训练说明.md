

# CCPD 车牌识别训练与导出说明

本 README 总结了在 PaddleOCR 上基于 CCPD 数据集训练车牌**识别**模型**（PP-OCRv5_rec)**的完整流程，包括：

- 环境配置
- 数据准备与标注转换
- 新增/修改的代码及其在仓库中的位置
- 训练与导出
- 模型评估效果

#### 相关文件总览：

##### 训练相关：

train_data文件夹，`_init_.py`、`ccpd_warp.py`、`make_gt.py`、`PP-OCRv5_server_rec_ccpd.yaml`

##### 模型评估相关：

`evaluate_train.py`、`evaluate_train_rec.py`

---

## 1. 环境与模型准备

在原先运行评估代码的环境下：

```bash
1、克隆原仓库：git clone https://github.com/PaddlePaddle/PaddleOCR.git
2、安装相关依赖：python -m pip install -r requirements.txt
3、下载预训练模型（与先前推理所用模型不同）：
wget -c https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams
```

---

## 2. 数据准备与标注转换

### 2.1 原始数据

- CCPD2019 ，额外添加CCPD2020的新增绿牌子集

- 划分文件（cby提供版本）：

  ```text
  /home/zwm/datasets/splits/all_train.txt
  /home/zwm/datasets/splits/all_test.txt   
  ```

`all_train.txt` 中每一行类似：

```text
ccpd_weather/0487-3_0-204&521_486&602-483&602_204&585_204&521_486&538-0_0_23_...jpg
```

### 2.2 标注转换脚本（生成识别用 txt 与字典）

**新增文件：**

- `tools/make_gt.py`

主要功能：

1. 根据 CCPD 的文件名规则，从文件名解码车牌文字（省份简称 + 字母 + 数字）。

2. 读取 `all_train.txt` / `all_test.txt` 等划分文件，生成 PaddleOCR 识别训练所需的 label 文件：
   - `train_data/rec/ccpd/rec_gt_test.txt`
   
   - `train_data/rec/ccpd/rec_gt_train.txt`
   
   - ```
     train.txt再随机划分出val与实际训练train
     
     shuf train_data/rec/ccpd/rec_gt_train_all.txt > /tmp/ccpd_train_shuf.txt
     
     TOTAL=$(wc -l < /tmp/ccpd_train_shuf.txt)
     VAL=$((TOTAL / 20))   # 随机取5%
     head -n $VAL /tmp/ccpd_train_shuf.txt > train_data/rec/ccpd/rec_gt_val.txt
     tail -n +$((VAL + 1)) /tmp/ccpd_train_shuf.txt > train_data/rec/ccpd/rec_gt_train.txt
     ```
   
3. 统计出现过的字符，生成车牌小字典：
   - `train_data/rec/ccpd/plate_dict.txt`

**生成的 label 文件格式：**

- 每行一条样本
- 通过 Tab (`\t`) 分隔，左边是图片绝对路径，右边是车牌字符，例如：

  ```text
  /home/zwm/datasets/ccpd_weather/0356-3_0-244&490_522&597-...jpg	皖AM295Y
  ```

---

## 3. 代码改动总览

| 文件路径                                              | 类型 | 作用                                                       |
| ----------------------------------------------------- | ---- | ---------------------------------------------------------- |
| `tools/make_gt.py`                                    | 新增 | 读取 CCPD 划分 txt，生成识别训练标注与字符字典             |
| `ppocr/data/imaug/ccpd_warp.py`                       | 新增 | 自定义增强算子 `CCPDWarp`，根据文件名中的 4 点做透视矫正   |
| `ppocr/data/imaug/__init__.py`                        | 修改 | `from .ccpd_warp import CCPDWarp`，让配置文件能调用该算子  |
| `configs/rec/PP-OCRv5/PP-OCRv5_server_rec_ccpd.yml`   | 修改 | 基于官方 PP-OCRv5 识别配置，替换为 CCPD 车牌任务相关的配置 |
| （生成文件）`train_data/rec/ccpd/rec_gt_train.txt` 等 | 生成 | PaddleOCR 训练/验证使用的 label 文件                       |
| （生成文件）`train_data/rec/ccpd/plate_dict.txt`      | 生成 | 本任务使用的车牌字符字典                                   |

---

## 4. 训练

### 4.1 相关日志

```bash
cd ~/work/PaddleOCR
conda activate ocr

python3 -u tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec_ccpd.yml   -o Global.pretrained_model=./pretrain/PP-OCRv5_server_rec_pretrained.pdparams      Global.save_model_dir=./output/ccpd_ppocrv5_server_rec      Global.epoch_num=10      Train.loader.num_workers=0      Eval.loader.num_workers=0   2>&1 | tee train_$(date +%F_%H%M).log
```
#### 1-10轮在验证集val的表现：

注意其正确率是仅评估识别文字效果，直接使用数据集标注的车牌框位置去识别车牌内容。


```
[2026/01/07 22:27:37] ppocr INFO: cur metric, acc: 0.05779889147725971, norm_edit_dis: 0.8111458502008816, fps: 1609.9560880669972

[2026/01/07 22:27:39] ppocr INFO: save best model is to ./output/ccpd_ppocrv5_server_rec/best_accuracy

[2026/01/07 22:27:39] ppocr INFO: best metric, acc: 0.05779889147725971, is_float16: False, norm_edit_dis: 0.8111458502008816, fps: 1609.9560880669972, best_epoch: 1
```

```
[2026/01/07 22:38:33] ppocr INFO: cur metric, acc: 0.8550189136491432, norm_edit_dis: 0.9777795931330024, fps: 1602.6777612381716

[2026/01/07 22:38:35] ppocr INFO: save best model is to ./output/ccpd_ppocrv5_server_rec/best_accuracy

[2026/01/07 22:38:35] ppocr INFO: best metric, acc: 0.8550189136491432, is_float16: False, norm_edit_dis: 0.9777795931330024, fps: 1602.6777612381716, best_epoch: 2
```

```
[2026/01/07 22:48:57] ppocr INFO: cur metric, acc: 0.962083222519501, norm_edit_dis: 0.9929849494770011, fps: 1606.4737067366448

[2026/01/07 22:48:59] ppocr INFO: save best model is to ./output/ccpd_ppocrv5_server_rec/best_accuracy

[2026/01/07 22:48:59] ppocr INFO: best metric, acc: 0.962083222519501, is_float16: False, norm_edit_dis: 0.9929849494770011, fps: 1606.4737067366448, best_epoch: 3
```

```
[2026/01/07 22:59:11] ppocr INFO: cur metric, acc: 0.9581243943361271, norm_edit_dis: 0.9924173079755755, fps: 1593.5156729322832

[2026/01/07 22:59:11] ppocr INFO: best metric, acc: 0.962083222519501, is_float16: False, norm_edit_dis: 0.9929849494770011, fps: 1606.4737067366448, best_epoch: 3
```

```
[2026/01/07 23:09:28] ppocr INFO: cur metric, acc: 0.9655142069450918, norm_edit_dis: 0.9936363757018843, fps: 1588.5604678772227

[2026/01/07 23:09:30] ppocr INFO: save best model is to ./output/ccpd_ppocrv5_server_rec/best_accuracy

[2026/01/07 23:09:30] ppocr INFO: best metric, acc: 0.9655142069450918, is_float16: False, norm_edit_dis: 0.9936363757018843, fps: 1588.5604678772227, best_epoch: 5
```

```
[2026/01/07 23:19:50] ppocr INFO: cur metric, acc: 0.9774786654548442, norm_edit_dis: 0.9955157717799437, fps: 1610.694218678236

[2026/01/07 23:19:52] ppocr INFO: save best model is to ./output/ccpd_ppocrv5_server_rec/best_accuracy

[2026/01/07 23:19:52] ppocr INFO: best metric, acc: 0.9774786654548442, is_float16: False, norm_edit_dis: 0.9955157717799437, fps: 1610.694218678236, best_epoch: 6
```

```
[2026/01/07 23:30:32] ppocr INFO: cur metric, acc: 0.9768628477374305, norm_edit_dis: 0.9957288991702391, fps: 1589.7854705350028

[2026/01/07 23:30:32] ppocr INFO: best metric, acc: 0.9774786654548442, is_float16: False, norm_edit_dis: 0.9955157717799437, fps: 1610.694218678236, best_epoch: 6
```

```
[2026/01/07 23:43:14] ppocr INFO: cur metric, acc: 0.983900764507873, norm_edit_dis: 0.9969755809449875, fps: 1582.5584233584493

[2026/01/07 23:43:16] ppocr INFO: save best model is to ./output/ccpd_ppocrv5_server_rec/best_accuracy

[2026/01/07 23:43:16] ppocr INFO: best metric, acc: 0.983900764507873, is_float16: False, norm_edit_dis: 0.9969755809449875, fps: 1582.5584233584493, best_epoch: 7
```

```
[2026/01/07 23:54:17] ppocr INFO: cur metric, acc: 0.9874197228930943, norm_edit_dis: 0.9973387877211763, fps: 1603.533644080144

[2026/01/07 23:54:18] ppocr INFO: save best model is to ./output/ccpd_ppocrv5_server_rec/best_accuracy

[2026/01/07 23:54:18] ppocr INFO: best metric, acc: 0.9874197228930943, is_float16: False, norm_edit_dis: 0.9973387877211763, fps: 1603.533644080144, best_epoch: 8
```

```
[2026/01/08 00:04:42] ppocr INFO: cur metric, acc: 0.9882994624893996, norm_edit_dis: 0.9976466965798833, fps: 1576.1415966416835

[2026/01/08 00:04:43] ppocr INFO: save best model is to ./output/ccpd_ppocrv5_server_rec/best_accuracy

[2026/01/08 00:04:43] ppocr INFO: best metric, acc: 0.9882994624893996, is_float16: False, norm_edit_dis: 0.9976466965798833, fps: 1576.1415966416835, best_epoch: 9
```


```
[2026/01/08 00:15:07] ppocr INFO: cur metric, acc: 0.9894431239645965, norm_edit_dis: 0.9978304993169684, fps: 1597.253040094435

[2026/01/08 00:15:08] ppocr INFO: save best model is to ./output/ccpd_ppocrv5_server_rec/best_accuracy

[2026/01/08 00:15:08] ppocr INFO: best metric, acc: 0.9894431239645965, is_float16: False, norm_edit_dis: 0.9978304993169684, fps: 1597.253040094435, best_epoch: 10
```

#### 测试最优模型在all_test数据集中的表现

```
python3 tools/eval.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec_ccpd.yml   -o Global.pretrained_model=./output/ccpd_ppocrv5_server_rec/latest      Eval.dataset.label_file_list=["train_data/rec/ccpd/rec_gt_test.txt"]
```

```
[2026/01/08 00:29:32] ppocr INFO: metric eval ***************
[2026/01/08 00:29:32] ppocr INFO: acc:0.993028167900546
[2026/01/08 00:29:32] ppocr INFO: norm_edit_dis:0.9983665923202735
[2026/01/08 00:29:32] ppocr INFO: fps:1409.3117010842625
```

#### 导出模型

训练完成后，使用 PaddleOCR 自带的 `export_model.py` 导出推理模型。

```bash
python3 tools/export_model.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec——ccpd.yml \
  -o Global.pretrained_model=./output/ccpd_ppocrv5_server_rec/best_accuracy \
     Global.save_inference_dir=./inference/ccpd_ppocrv5_server_rec
```

说明：

- `Global.pretrained_model` 指向想要导出的 checkpoint，一般用 `best_accuracy`。
  - 对应文件为：`./output/ccpd_ppocrv5_server_rec/best_accuracy.pdparams`
- `Global.save_inference_dir` 为导出的推理模型目录，例如：`./inference/ccpd_ppocrv5_server_rec`

导出后目录结构类似：

```text
./inference/ccpd_ppocrv5_server_rec/
├── inference.json
├── inference.pdiparams
└── inference.yml
```

---

由于导出模型是用来做端到端评测，检测发现

 `inference.yml` 的 `PreProcess.transform_ops` 里混进了**训练用算子**（`CCPDWarp`、`MultiLabelEncode`、`KeepKeys` 里还保留了 label_xxx），端到端推理/评估时不该出现这些，不符合“不能用真值框/真值透视”的要求。

故修改 `PreProcess.transform_ops` ：

核心是**删掉 CCPDWarp、删掉 MultiLabelEncode、KeepKeys 只保留 image**：

```
PreProcess:
  transform_ops:
  - DecodeImage:
      channel_first: false
      img_mode: BGR
  - RecResizeImg:
      image_shape: [3, 48, 320]
  - KeepKeys:
      keep_keys: [image]
```

## 5. 模型评估效果

##### 模型评估相关

本节主要基于 CCPD2020 Test 集，对微调后的识别模型进行两类评估：

- `evaluate_train.py`：**端到端车牌检测 + 识别评估**
- `evaluate_train_rec.py`：**仅识别（rec）评估**

### 5.1 端到端评估：`evaluate_train.py`

该脚本复现了“真实应用场景”下的完整流程：  
首先使用 PaddleOCR 的检测模型在整图中定位车牌区域，然后使用**训练后的 PP-OCRv5 识别模型**完成字符识别，并统计整牌与字符级指标。

- 数据集：all_test.txt指定图片

- 主要指标：
  
  ```
  ==================== 最终评估报告 ====================
  错误样本清单(bad_cases): /home/zwm/ocr/e2e_eval_out/bad_cases.jsonl
  评测图片数量: 135402
  
  1. 准确率 (Accuracy)
  
     - 全字匹配率（整牌全对）: 90.04%
     - 字符准确率（逐字符平均）: 94.05%
  
  2. 推理速度 (Latency)
  
     - 平均单张耗时: 49.08 ms
     - FPS（每秒处理张数）: 20.38
  
  3. 鲁棒性 (Robustness)
  
     - all_hardtest 子集整牌匹配率: 64.87%   (样本数=30400)
       ======================================================
  
  错误可视化输出目录: /home/zwm/ocr/e2e_eval_out/vis_err
  错误可视化保存张数: 50 / 50
  ```

> 注：样本太多端到端检测评估耗时太长了，暂时就只跑了训练后的效果。
>
> 说明：端到端结果受“检测 + 识别”两个环节共同影响，90% 左右的整牌正确率表明，在现有检测器和微调识别模型的组合下，大部分车牌已经可以被完整地检测并正确识别。

#### 5.1.1 端到端失败案例示例：

> 由可视化结果发现漏框车牌框占比相当一半

---

### 5.2 仅识别评估：`evaluate_train_rec.py`

该脚本只评估**识别模型本身的能力**：  
直接使用 CCPD2020 文件名编码的车牌真值框裁剪出车牌区域，将其输入**微调后的 PP-OCRv5 识别模型**，不再引入检测误差。

- 数据集：all_test.txt指定图片

- 主要指标：

- ```
  用训练后的模型识别
  ==================== 最终评估报告（Rec-only / GT-Warp）====================
  错误样本清单(bad_cases): /home/zwm/ocr/warp_rec_only_out/bad_cases_warp.jsonl
  评测图片数量: 135402
  
  1. 准确率 (Accuracy)
  
     - 全字匹配率（整牌全对）: 99.30%
     - 字符准确率（逐字符平均）: 99.78%
  
  2. 推理速度 (Latency)
  
     - 平均单张耗时: 6.70 ms
     - FPS（每秒处理张数）: 149.22
  
  3. 鲁棒性 (Robustness)
  
     - all_hardtest 子集整牌匹配率: 98.21%   (样本数=30400)
       ======================================================
  
  错误可视化输出目录: /home/zwm/ocr/warp_rec_only_out/vis_err_50
  错误可视化保存张数: 50 / 50
  ```
  
- ```
  用初始的模型识别
  ==================== 最终评估报告（Rec-only / GT-Warp）====================
  错误样本清单(bad_cases): /home/zwm/ocr/warp_rec_only_out/bad_cases_warp.jsonl
  评测图片数量: 135402
  
  1. 准确率 (Accuracy)
  
     - 全字匹配率（整牌全对）: 78.88%
     - 字符准确率（逐字符平均）: 88.44%
  
  2. 推理速度 (Latency)
  
     - 平均单张耗时: 7.85 ms
     - FPS（每秒处理张数）: 127.38
  
  3. 鲁棒性 (Robustness)
  
     - all_hardtest 子集整牌匹配率: 45.23%   (样本数=30400)
       ======================================================
  
  错误可视化输出目录: /home/zwm/ocr/warp_rec_only_out/vis_err_50
  错误可视化保存张数: 50 / 50
  ```
  
- 评估方式：
  
  - 每张图片先根据文件名解析出车牌的四点坐标；
  - 进行裁剪 / 透视矫正后送入识别模型；

> 对比训练前的模型，全字符匹配率提升20%， 鲁棒性提升53%。
>
> 说明：在理想的“真值车牌框”条件下，训练后的模型可以达到约 **99%** 的整牌正确率，说明识别模型对车牌字符分布已经高度适应；与端到端评估的差距主要来自**车牌检测环节的误差**。

#### 5.2.1 仅识别失败案例示例

> 老问题，Q、O、D与0混淆，过模糊或遮挡导致识别失败
>

### 5.3 视频检测效果：

![image-20260108115145779](C:\Users\35011\AppData\Roaming\Typora\typora-user-images\image-20260108115145779.png)

![image-20260108115238186](C:\Users\35011\AppData\Roaming\Typora\typora-user-images\image-20260108115238186.png)

图一为训练后图二为先前

对比结果虽然仍存在部分字误识别（极少数），但个人觉得效果提升显著，检测视频大多为俯拍，有多帧，故而PP-OCRv5找不到检测框的概率较低，因此总体效果较好。

---

## 6. 未来改进方向

当前实验表明：  
- 微调后的识别模型在“给定真值框”的前提下已有接近 **99%** 的整牌正确率；  
- 端到端准确率主要受**车牌检测**与**极端模糊样本**的限制。

后续可以从以下几个方向进一步改进：

#### 方案 A：训练/微调 PaddleOCR 的 det（DB 系列）做“车牌检测”

##### 优点

和 PaddleOCR pipeline 天然兼容

##### 缺点

DB 是“文本区域”检测，不是“车牌目标检测”。在复杂场景里更容易：拆框/多框/局部框 ，通常要花很多精力在阈值、后处理规则、数据增强和失败案例打磨上。


#### 方案 B：用 YOLO 训练车牌 detector

端到端掉分本质是几何没对齐，而 YOLO 可以直接输出需要的角点，从源头解决“裁剪质量”问题。


##### 方案 C：不训练模型，先用“Top-K 候选框 + 识别置信度/正则投票”

端到端 90%，很可能不是 det 完全不行，而是“det 给了多个文本框，挑错了”。

做法：

det 阶段保留 top-K（比如 3~10 个）候选框（不要只取一个）
对每个候选框都跑一次 rec
用规则选最合理的：
优先满足车牌正则（已有）
其次看 rec_score
再加几何先验

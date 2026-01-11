import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from imutils import paths
import cv2
import os

# ==========================================
# 1. 定义字符映射表 (根据你提供的信息)
# ==========================================
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O']
ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

# LPRNet 训练需要的全局字符表（去重并排序，保持 'O' 作为空白符在最后或者是特定的位置）
# 这里我们构建一个包含所有可能字符的列表
CHARS = ['皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', "青", "宁", "新", "警", "学", 
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'] # 最后加个 '-' 作为空白符(blank)

# 创建字符到索引的字典，方便转换
CHAR_DICT = {char: i for i, char in enumerate(CHARS)}

# ==========================================
# 2. 自定义 CCPD 数据集读取类
# ==========================================
class CCPDDataset(Dataset):
    def __init__(self, img_paths, img_size=(94, 24), transform=None):
        self.img_paths = img_paths
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        image = cv2.imread(filename)
        
        # 异常处理：如果读图失败
        if image is None:
            return self.__getitem__(np.random.randint(self.__len__()))

        h, w, _ = image.shape
        
        # --- 解析文件名 ---
        # 示例: 025-95_113-154,383_386,473-386,473_177,454_154,383_363,402-0_0_22_27_27_33_16-37-15.jpg
        try:
            basename = os.path.basename(filename)
            split_name = basename.split('-')
            
            # 1. 获取边界框 (Bounding Box) - 对应索引 2
            # 格式: 154&383_386&473 (LeftUp_RightBottom) -> x1&y1_x2&y2
            # --- 修正开始 ---
            coords = split_name[2].split('_')
            
            # 1. 先把可能出现的逗号替换成 '&'
            # 2. 然后再 split，这样无论数据是 "100&200" 还是 "100,200" 都能跑
            txt_point1 = coords[0].replace(',', '&')
            txt_point2 = coords[1].replace(',', '&')
            
            x1, y1 = map(int, txt_point1.split('&'))
            x2, y2 = map(int, txt_point2.split('&'))
            # --- 修正结束 ---
            
            # 修正坐标防止越界
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 裁剪图片 (这里实现了你想要的：只用 Box，不用透视变换)
            crop_img = image[y1:y2, x1:x2]
            
            # 防止空裁剪
            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                raise ValueError("Empty crop")

            # 2. 解析 Label - 对应索引 4
            # 格式: 0_0_22_27_27_33_16 (Indexes in Provinces, Alphabets, Ads)
            lbl_indices = split_name[4].split('_')
            label_str = []
            
            # CCPD 规则: 
            # 第1位: Province
            label_str.append(PROVINCES[int(lbl_indices[0])])
            # 第2位: Alphabet
            label_str.append(ALPHABETS[int(lbl_indices[1])])
            # 第3-7位: Ads (Alphabet + Digits)
            for i in range(2, 7):
                label_str.append(ADS[int(lbl_indices[i])])
            
            # 将汉字/字符转换为全局 CHARS 的索引
            label = [CHAR_DICT[c] for c in label_str]
            label = np.array(label, dtype=np.int32)
            
            # 3. 图片预处理 (Resize -> Normalize -> Transpose)
            # LPRNet 标准输入是 (94, 24)
            crop_img = cv2.resize(crop_img, self.img_size)
            # 归一化到 [-1, 1] 或者是 [0, 1]，LPRNet 原版习惯减 127.5
            crop_img = crop_img.astype('float32')
            crop_img -= 127.5
            crop_img *= 0.0078125
            crop_img = np.transpose(crop_img, (2, 0, 1)) # HWC -> CHW

            return torch.from_numpy(crop_img), torch.from_numpy(label), len(label)

        except Exception as e:
            # print(f"Error processing {filename}: {e}")
            # 出错就换一张图读，保证训练不中断
            return self.__getitem__(np.random.randint(self.__len__()))

print(f"✅ 全局字符表长度: {len(CHARS)}")
print(f"示例字符表: {CHARS[:10]} ...")

# 修改这里的 ROOT_PATH 为你实际挂载的路径
# 你可以通过 ls 命令查看： !ls /kaggle/input/
DATASET_ROOT = "/kaggle/input/ccpd-preprocess/CCPD2019"  # <--- 请根据实际情况修改这里！！
SPLIT_FOLDER = os.path.join(DATASET_ROOT, "splits")

def get_image_paths(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    # 拼接完整路径
    return [os.path.join(DATASET_ROOT, line.strip()) for line in lines]

# 读取 all_train.txt 和 all_test.txt
train_txt = os.path.join(SPLIT_FOLDER, "all_train.txt")
test_txt = os.path.join(SPLIT_FOLDER, "all_test.txt")

# 检查文件是否存在，如果路径不对，请手动调整 DATASET_ROOT
if not os.path.exists(train_txt):
    print(f"❌ 找不到 split 文件: {train_txt}")
    print("请使用 !find /kaggle/input -name 'all_train.txt' 查找真实路径")
else:
    train_paths = get_image_paths(train_txt)
    test_paths = get_image_paths(test_txt)
    
    print(f"✅ 训练集加载: {len(train_paths)} 张")
    print(f"✅ 测试集加载: {len(test_paths)} 张")

    # 构建 DataLoader
    train_dataset = CCPDDataset(train_paths)
    val_dataset = CCPDDataset(test_paths)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    
sys.path.append('/kaggle/working/LPRNet_Pytorch')
from model.LPRNet import LPRNet

# 修改 LPRNet 初始化代码
# lpr_max_len=8: 车牌最大长度（CCPD是7位，一般设为8预留一位或作为标准）
# phase=True: 表示当前是训练阶段 (会启用 Dropout)
lprnet = LPRNet(lpr_max_len=8, phase=True, class_num=len(CHARS), dropout_rate=0.5)

lprnet = lprnet.cuda()

# ============================================================
# 🟢 新增/修改代码: 加载断点权重 (Resume Training)
# ============================================================
# 这里填写你想要加载的权重文件路径
# 如果是同一环境未重启，路径通常是 '/kaggle/working/weights/lprnet_best.pth'
# 如果你重启了环境，你需要上传之前的权重并修改这里的路径
RESUME_WEIGHT_PATH = '/kaggle/input/lprnet/pytorch/default/1/lprnet_epoch_3.pth' 

if os.path.exists(RESUME_WEIGHT_PATH):
    print(f"🔄 发现预训练权重: {RESUME_WEIGHT_PATH}")
    # 加载权重
    lprnet.load_state_dict(torch.load(RESUME_WEIGHT_PATH))
    print("✅ 权重加载成功！将在该基础上继续训练。")
    
    # 【可选】如果你知道之前的最佳准确率（例如 85%），可以手动设置，防止刚开始训练效果不好把好模型覆盖了
    # best_acc = 0.85 
else:
    print("⚠️ 未找到权重文件，将从头开始训练 (Start from scratch)。")
# ============================================================

print("✅ LPRNet 模型初始化完成")
print(lprnet)

# 看看 STN 是否开启 (LPRNet 默认带 STN)
print(lprnet)

import os

# 1. 根据你的截图，这是绝对正确的根目录路径
DATASET_ROOT = "/kaggle/input/ccpd-preprocess/CCPD2019"

def get_image_paths(txt_file):
    valid_paths = []
    
    # 检查 split 文件是否存在
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"找不到索引文件: {txt_file}")
        
    with open(txt_file, 'r') as f:
        lines = f.readlines()
        
    print(f"正在处理 {os.path.basename(txt_file)}，共 {len(lines)} 行...")
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # --- 路径清洗逻辑 (关键步骤) ---
        # 原始行可能是: /tmp/CCPD2019/ccpd_base/xxx.jpg
        # 我们只需要: ccpd_base/xxx.jpg
        
        if "CCPD2019/" in line:
            # 这里的 split 会把路径切成两半，我们取后面那半
            # 例如: ['', 'ccpd_base/xxx.jpg']
            rel_path = line.split("CCPD2019/")[-1]
        else:
            # 如果路径里居然没有 CCPD2019，那就假设它已经是相对路径了
            rel_path = line
            
        # 去掉开头可能存在的斜杠，防止 os.path.join 失效
        if rel_path.startswith('/'):
            rel_path = rel_path[1:]
            
        # 拼接成 Kaggle 的真实路径
        full_path = os.path.join(DATASET_ROOT, rel_path)
        valid_paths.append(full_path)

    # --- 验证逻辑：检查第一张图能不能找到 ---
    if valid_paths:
        first_img = valid_paths[0]
        if not os.path.exists(first_img):
            print(f"❌ 路径修正失败！请检查！")
            print(f"原始文本: {lines[0].strip()}")
            print(f"修正后路径: {first_img}")
            print(f"期望的根目录: {DATASET_ROOT}")
            raise FileNotFoundError("无法找到图片文件，路径拼接有误。")
        else:
            print(f"✅ 路径修正成功！")
            print(f"示例: {first_img}")
            
    return valid_paths

# 2. 重新加载路径 (指向 splits 文件夹)
train_txt = os.path.join(DATASET_ROOT, "splits/all_train.txt")
test_txt = os.path.join(DATASET_ROOT, "splits/all_test.txt")

try:
    train_paths = get_image_paths(train_txt)
    test_paths = get_image_paths(test_txt)

    # 3. 只有路径加载成功后，才重建 DataLoader
    # (你需要确保之前定义过 CCPDDataset 类)
    train_dataset = CCPDDataset(train_paths)
    val_dataset = CCPDDataset(test_paths)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    print("\n🎉 数据集加载完毕，现在可以重新运行训练代码块了！")

except Exception as e:
    print(f"\n❌ 发生错误: {e}")
    
import torch
import torch.nn as nn
import torch.optim as optim
import os

# ================= 配置 =================
EPOCHS = 5
LEARNING_RATE = 0.001 
SAVE_DIR = '/kaggle/working/weights/'
best_acc = 0.0

# 确保权重目录存在
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 1. 定义 Loss (CTCLoss)
# blank=len(CHARS)-1 表示使用我们在 CHARS 列表最后加的那个 '-' 作为空白符
ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') 

# 2. 定义优化器
optimizer = optim.Adam(lprnet.parameters(), lr=LEARNING_RATE)

# 2.5 定义学习率调度器 - 余弦退火
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# 3. 解码函数 (用于计算准确率)
def greedy_decode(preds):
    preds = preds.argmax(dim=2)
    preds = preds.detach().cpu().numpy()
    res = []
    for i in range(preds.shape[1]): # batch size
        temp = []
        for k in range(preds.shape[0]): # time steps
            # 如果不是 blank 且 (是第一个字符 OR 与前一个字符不同) -> 保存
            if preds[k, i] != len(CHARS)-1 and (k==0 or preds[k, i] != preds[k-1, i]):
                temp.append(preds[k, i])
        res.append(temp)
    return res

# ================= 训练主循环 =================
print(f"🚀 开始训练... 目标轮数: {EPOCHS}")

for epoch in range(EPOCHS):
    lprnet.train()
    loss_val = 0
    
    # --- Training ---
    for i, (imgs, labels, lengths) in enumerate(train_loader):
        imgs = imgs.cuda()
        labels = labels.cuda() # 训练时 Label 要上 GPU 配合模型
        
        # LPRNet 输出的时间步长固定是 18 (针对 94x24 的输入)
        input_lengths = (torch.ones(imgs.size(0)) * 18).int() 
        # CCPD 车牌固定长度 7
        target_lengths = torch.tensor([7] * imgs.size(0)).int() 
        
        # 将 batch 的 label 展平以适配 CTCLoss
        targets = []
        for label in labels:
            targets.extend(label.tolist())
        targets = torch.tensor(targets).int()
        
        # 前向传播
        optimizer.zero_grad()
        logits = lprnet(imgs)      # [batch, class_num, 18]
        logits = logits.permute(2, 0, 1) # [18, batch, class_num]
        logits = logits.log_softmax(2)
        
        # 计算 Loss
        loss = ctc_loss(logits, targets, input_lengths, target_lengths)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        loss_val += loss.item()
        
        if i % 50 == 0: # 每50个batch打印一次
            print(f"Epoch [{epoch+1}/{EPOCHS}] Iter [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")

    # --- Validation ---
    lprnet.eval()
    correct = 0
    total = 0
    print(f"🔍 正在验证第 {epoch+1} 轮模型...")
    
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs = imgs.cuda()
            # 注意：这里 labels 不需要 .cuda()，因为后面的对比逻辑是在 CPU 上进行的
            
            logits = lprnet(imgs)
            logits = logits.permute(2, 0, 1)
            
            # 解码
            preds = greedy_decode(logits)
            
            # 对比真值
            for j in range(len(preds)):
                pred_label = preds[j]
                # labels 是 DataLoader 出来的 CPU Tensor
                true_label = labels[j].numpy().tolist()
                
                if pred_label == true_label:
                    correct += 1
                total += 1
    
    acc = correct / total
    print(f"🏆 Epoch {epoch+1} 验证准确率: {acc*100:.2f}%")
    
    # 保存最佳模型
    if acc > best_acc:
        best_acc = acc
        save_path = os.path.join(SAVE_DIR, f'lprnet_best.pth') # 保存一个固定名字方便下载
        torch.save(lprnet.state_dict(), save_path)
        print(f"🔥 新纪录！最佳模型已保存: {save_path}")
    
    # 也可以保存这一轮的 checkpoint (可选)
    torch.save(lprnet.state_dict(), os.path.join(SAVE_DIR, 'lprnet_last.pth'))
    
    # 保存每一轮 (增量式保存，用于历史回溯) <--- 这里是你想要的
    epoch_path = os.path.join(SAVE_DIR, f'lprnet_epoch_{epoch+1}.pth')
    torch.save(lprnet.state_dict(), epoch_path)
    print(f"📂 已归档本轮权重: {epoch_path}")
    
    # 更新学习率（余弦退火）
    scheduler.step()
    print(f"📊 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
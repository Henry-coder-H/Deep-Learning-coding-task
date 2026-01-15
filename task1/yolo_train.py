# train.py
from ultralytics import YOLO
import torch

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 选择模型
    model = YOLO("yolo11l.pt") 

    # 开始训练
    results = model.train(
        data='/data2/zhuangyn/Deep-Learning-coding-task/task1/dataset/BIT_YOLO_Dataset/data.yaml',       # 数据配置
        epochs=100,
        batch=32,                           # 根据 GPU 显存调整（32G 可设 64）
        patience=10,                        # 早停机制
        # imgsz=640,                        # BIT-Vehicle 图像多为中等目标，640 足够
        imgsz=1088,                         # 1088x1088，适配 BIT 的 1080p 高清图，大幅提升小目标检测率
        
        name='bit_vehicle_yolo11_train',
        device="cuda:0,1,2,3",
        workers=16,
        save=True,
        exist_ok=False,        # 不覆盖旧实验


        # 数据增强（应对城市道路复杂场景）
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=10.0,               # ±10° 旋转（模拟不同拍摄角度）
        translate=0.1,
        scale=0.4,
        flipud=0.0,                 # 不上下翻转（车辆方向敏感）
        fliplr=0.5,                 # 左右翻转安全
        mosaic=1.0,                 # Mosaic 增强（提升小车检出率）
        mixup=0.1,
        # copy_paste=0.2,           # 对少数类（如 Minivan）有效

        plots=True                  # 自动画出混淆矩阵、PR曲线等
    )

    # 验证集评估
    metrics = model.val(data='/Deep-Learning-coding-task/task1/dataset/BIT_YOLO_Dataset/data.yaml', split='val') # 修改为对应的数据集路径
    print("\n✅ BIT-Vehicle 验证集结果:")
    print(f"  mAP@0.5:         {metrics.box.map50:.5f}")
    print(f"  mAP@0.5:0.95:    {metrics.box.map:.5f}")
    print(f"  Precision:       {metrics.box.mp:.5f}")
    print(f"  Recall:          {metrics.box.mr:.5f}")

if __name__ == '__main__':
    main()

# 模型权重目录

请将以下权重文件放置在此目录：

## 必需的权重文件

1. **plate_detect.pt** - 车牌检测模型 (YOLOv8-Pose)
   - 来源: `task2_member1/runs/pose/train/weights/best.pt`
   - 复制命令: `cp ../task2_member1/runs/pose/train/weights/best.pt plate_detect.pt`

2. **plate_rec.pth** - 车牌识别模型 (ResNet-CRNN)
   - 来源: `task2_member1/runs/rec/best.pth`
   - 复制命令: `cp ../task2_member1/runs/rec/best.pth plate_rec.pth`

3. **yolov11l.pt** - 车辆检测模型 (YOLOv11l)
   - 来源: 从 Ultralytics 官网下载 `yolo11l.pt`
   - 下载后重命名为 `yolov11l.pt`

4. **`vehicle_classifier.pt** - 车型分类模型
   - 来源：`task1/best.pth`
   - 复制命令: `cp task1/best.pth UI_App/weights/vehicle_classifier.pt`

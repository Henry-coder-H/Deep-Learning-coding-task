import argparse
from pathlib import Path

from ultralytics import YOLO


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 pose for plate corners.")
    parser.add_argument("--data", type=str, default="configs/data_ccpd_kpts.yaml")
    parser.add_argument("--pretrained", type=str, default="yolov8n-pose.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=str, default="runs/pose")
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = build_args()
    model = YOLO(args.pretrained)
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Data config not found: {data_path}")

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        lr0=args.lr,
        workers=args.workers,
    )
    print(f"[INFO] start training with args: {train_kwargs}")
    model.train(**train_kwargs)
    print("[INFO] training done. Best weights in runs/pose/*/weights/best.pt")


if __name__ == "__main__":
    main()


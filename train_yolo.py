import torch
from ultralytics import YOLO

def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Training on: {gpu_name}")

    model = YOLO("yolov8l.pt")

    results = model.train(
        data="coco128.yaml", epochs=100, imgsz=640,
        batch=16, device=device, workers=8,
        optimizer="AdamW", lr0=0.001, lrf=0.01,
        mosaic=1.0, mixup=0.1, copy_paste=0.1,
        amp=True, project="runs/detect", name="yolov8_amd"
    )

    # Export to ONNX for production inference
    model.export(format="onnx", dynamic=True, simplify=True)
    print("Model exported to ONNX")

if __name__ == "__main__":
    main()

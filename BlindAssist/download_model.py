#!/usr/bin/env python3
import os, sys, shutil

def main():
    print("=" * 50)
    print("  Blind Assist - Model Downloader")
    print("=" * 50)
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO

    print("\nDownloading YOLOv8n...")
    model = YOLO("yolov8n.pt")
    print("Converting to ONNX...")
    model.export(format="onnx", imgsz=640, simplify=True, opset=12)

    dst_dir = os.path.join("app", "src", "main", "assets")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "yolov8n.onnx")

    if os.path.exists("yolov8n.onnx"):
        shutil.copy("yolov8n.onnx", dst)
        print(f"\n  Model saved to: {dst}")
        print("  Open this folder in Android Studio and press Run.")
    else:
        print("\n  ERROR: yolov8n.onnx not found. Try running again.")
    print("=" * 50)

if __name__ == "__main__":
    main()

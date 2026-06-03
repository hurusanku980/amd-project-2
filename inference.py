import cv2
import torch
import time
from ultralytics import YOLO

def run_inference(video_source=0, model_path="runs/detect/yolov8_amd/weights/best.pt"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)

    fps_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model(frame, verbose=False)
        fps = 1.0 / (time.time() - start)
        fps_list.append(fps)

        annotated = results[0].plot()
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLOv8 AMD Inference", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Average FPS: {sum(fps_list)/len(fps_list):.1f}")

if __name__ == "__main__":
    run_inference()

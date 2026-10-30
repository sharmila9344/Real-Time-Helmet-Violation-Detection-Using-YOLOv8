from ultralytics import YOLO
import os

# Path to helmet detection model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "best.pt"
)

# Load model
model = YOLO(MODEL_PATH)


def detect_helmet(image_path):
    """
    Detect helmet / no helmet in an image.
    """

    results = model.predict(
        source=image_path,
        conf=0.30,
        imgsz=640,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        detections.append({
            "class": class_name,
            "confidence": confidence,
            "box": box.xyxy[0].tolist()
        })

    return detections
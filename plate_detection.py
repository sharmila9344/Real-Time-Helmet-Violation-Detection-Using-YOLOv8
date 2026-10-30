from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2


# ============================================================
# HUGGING FACE LICENSE PLATE MODEL
# ============================================================

MODEL_REPO = "morsetechlab/yolov11-license-plate-detection"
MODEL_FILE = "license-plate-finetune-v1n.pt"

MODEL_PATH = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)

model = YOLO(MODEL_PATH)


# ============================================================
# LICENSE PLATE DETECTION
# ============================================================

def detect_plate(image_path, conf_threshold=0.05):
    """
    Detect and crop a license plate.

    A low confidence threshold is used because
    small or distant plates can have low detection confidence.
    """

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        return None

    height, width = image.shape[:2]


    # --------------------------------------------------------
    # Run YOLO
    # --------------------------------------------------------

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        imgsz=1280,
        verbose=False
    )

    boxes = results[0].boxes


    # --------------------------------------------------------
    # Check detections
    # --------------------------------------------------------

    if boxes is None or len(boxes) == 0:
        return None


    detections = []


    # --------------------------------------------------------
    # Collect detections
    # --------------------------------------------------------

    for box in boxes:

        xyxy = box.xyxy[0].cpu().numpy()

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = xyxy

        # Convert to integers
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(width, x2))
        y2 = int(min(height, y2))

        box_width = x2 - x1
        box_height = y2 - y1

        # Ignore extremely tiny boxes
        if box_width < 10 or box_height < 5:
            continue

        detections.append({
            "box": [x1, y1, x2, y2],
            "confidence": confidence
        })


    if not detections:
        return None


    # ========================================================
    # MERGE NEARBY PLATE DETECTIONS
    # ========================================================

    merged = []

    # Sort from left to right
    detections.sort(
        key=lambda d: d["box"][0]
    )


    current_box = detections[0]["box"]

    current_confidence = detections[0]["confidence"]


    for detection in detections[1:]:

        x1, y1, x2, y2 = detection["box"]

        cx1, cy1, cx2, cy2 = current_box


        # Horizontal gap
        horizontal_gap = x1 - cx2


        # Vertical overlap
        overlap_y = min(cy2, y2) - max(cy1, y1)


        # Merge if boxes are close
        if horizontal_gap <= 80 and overlap_y >= -30:

            current_box = [
                min(cx1, x1),
                min(cy1, y1),
                max(cx2, x2),
                max(cy2, y2)
            ]

            current_confidence = max(
                current_confidence,
                detection["confidence"]
            )

        else:

            merged.append({
                "box": current_box,
                "confidence": current_confidence
            })

            current_box = detection["box"]

            current_confidence = detection["confidence"]


    # Add last box
    merged.append({
        "box": current_box,
        "confidence": current_confidence
    })


    # ========================================================
    # SELECT BEST PLATE
    # ========================================================

    # Prefer wider boxes because Indian number plates
    # are normally wider than they are tall.

    def plate_score(item):

        x1, y1, x2, y2 = item["box"]

        box_width = x2 - x1
        box_height = y2 - y1

        if box_height == 0:
            return 0

        aspect_ratio = box_width / box_height

        # Favor plate-like aspect ratios
        aspect_bonus = min(aspect_ratio / 4.0, 1.0)

        return (
            item["confidence"] * 0.7
            + aspect_bonus * 0.3
        )


    best = max(
        merged,
        key=plate_score
    )


    x1, y1, x2, y2 = best["box"]


    # ========================================================
    # EXPAND CROP
    # ========================================================

    box_width = x2 - x1
    box_height = y2 - y1


    x1 = max(
        0,
        int(x1 - box_width * 0.10)
    )

    x2 = min(
        width,
        int(x2 + box_width * 0.10)
    )

    y1 = max(
        0,
        int(y1 - box_height * 0.60)
    )

    y2 = min(
        height,
        int(y2 + box_height * 0.60)
    )


    # ========================================================
    # CROP PLATE
    # ========================================================

    plate_crop = image[
        y1:y2,
        x1:x2
    ]


    if plate_crop.size == 0:
        return None


    # ========================================================
    # RETURN
    # ========================================================

    return {
        "crop": plate_crop,
        "confidence": best["confidence"],
        "box": [x1, y1, x2, y2]
    }
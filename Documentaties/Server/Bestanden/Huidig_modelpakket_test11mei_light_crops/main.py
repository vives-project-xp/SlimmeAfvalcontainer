import cv2
import time
from detector import YOLODetector
from classifier import TwoStageGarbageClassifier

# --- Configuration ---
DETECTOR_MODEL  = "yolov8_detector.onnx"
# We expect the current directory to contain the ONNX models and metadata
CLASSIFIER_DIR   = "."
CAMERA_INDEX     = 0
RESOLUTION       = (640, 480)
MIN_BOX_AREA     = 2000    # ignore tiny detections (noise)
MAX_BOX_AREA_RATIO = 0.85  # ignore near full-frame detections

# Colors for bounding boxes
CATEGORY_COLORS = {
    "Organisch":   (0,   165, 255),
    "PMD":         (255, 200,   0),
    "Papier":      (100, 200, 100),
    "Restafval":   (80,   80,  80),
    "Overige":     (180, 180, 180),
    "Elektronica": (0,   100, 255),   # blauw
    "Glas":        (180, 255, 255),   # lichtcyaan
}

detector   = YOLODetector(
    DETECTOR_MODEL,
    conf_threshold=0.4,
    iou_threshold=0.35,
    min_box_area_ratio=0.002,
    max_box_area_ratio=MAX_BOX_AREA_RATIO,
    max_aspect_ratio=8.0,
    max_detections=6,
)
classifier = TwoStageGarbageClassifier(model_dir=CLASSIFIER_DIR, stage1_threshold=0.60, stage2_threshold=0.55)

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

print("Running -- press Q to quit")

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)

    for (x1, y1, x2, y2, det_conf, _) in detections:
        # Skip tiny detections and near full-frame detections
        area = (x2 - x1) * (y2 - y1)
        if area < MIN_BOX_AREA:
            continue
        if area / max(1.0, frame.shape[0] * frame.shape[1]) > MAX_BOX_AREA_RATIO:
            continue

        # Add a small padding around the crop
        pad = 8
        crop = frame[
            max(0, y1 - pad): min(frame.shape[0], y2 + pad),
            max(0, x1 - pad): min(frame.shape[1], x2 + pad)
        ]

        label, clf_conf = classifier.classify(crop)
        
        # Determine color based on main category
        main_cat = label.split('/')[0] if '/' in label else label
        color = CATEGORY_COLORS.get(main_cat, (200, 200, 200))

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background + text
        text = f"{label}  {clf_conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    fps = 1 / (time.time() - t0)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Garbage detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

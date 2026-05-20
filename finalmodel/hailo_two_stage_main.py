import cv2
import time
import os
import numpy as np

from classifier import TwoStageGarbageClassifier

# ── Configuratie ─────────────────────────────────────────────────────────────
HEF_MODEL_PATH = "yolov8_detector.hef"  # Het gecompileerde Hailo model
CLASSIFIER_DIR = "."                    # Map met stage1_main.onnx, stage2_overige.onnx en metadata
RESOLUTION     = (1280, 720)
MIN_BOX_AREA   = 3000
MAX_BOX_AREA_RATIO = 0.85
CONF_THRESHOLD = 0.45
NMS_IOU_THRESHOLD = 0.35
MAX_DETECTIONS = 6

CATEGORY_COLORS = {
    "Organisch":   (0,   165, 255),
    "PMD":         (255, 200,   0),
    "Papier":      (100, 200, 100),
    "Restafval":   (80,   80,  80),
    "Overige":     (180, 180, 180),
    "Elektronica": (0,   100, 255),
    "Glas":        (180, 255, 255),
    "Batterijen":  (0,   0,   255),
    "Metaal":      (200, 200, 200),
    "Lightbulbs":  (255, 255, 0),
}

# ── Hailo / PiCamera2 Import ────────────────────────────────────────────────
try:
    from picamera2 import Picamera2
    from picamera2.devices.hailo import Hailo
    HAILO_OK = True
except ImportError:
    HAILO_OK = False
    print("Fout: picamera2 of hailo niet gevonden op dit systeem.")


def parse_hailo_output(hailo_output, orig_w: int, orig_h: int):
    detections = []
    if hailo_output is None:
        return detections

    for detection in hailo_output:
        label_id  = int(detection.get_label())
        conf      = float(detection.get_confidence())
        if conf < CONF_THRESHOLD:
            continue
        bbox = detection.get_bbox()
        x1 = int(bbox.xmin() * orig_w)
        y1 = int(bbox.ymin() * orig_h)
        x2 = int(bbox.xmax() * orig_w)
        y2 = int(bbox.ymax() * orig_h)
        bw, bh = x2 - x1, y2 - y1
        if bw * bh < MIN_BOX_AREA:
            continue
        if (bw * bh) / (orig_w * orig_h) > MAX_BOX_AREA_RATIO:
            continue
        detections.append((x1, y1, x2, y2, conf, label_id))
    return apply_nms(detections)


def apply_nms(detections):
    if not detections:
        return []

    boxes = []
    scores = []
    for x1, y1, x2, y2, conf, _label_id in detections:
        boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
        scores.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_IOU_THRESHOLD)
    if indices is None or len(indices) == 0:
        return []

    kept = np.array(indices).reshape(-1).astype(int).tolist()
    kept.sort(key=lambda idx: detections[idx][4], reverse=True)
    kept = kept[:MAX_DETECTIONS]
    return [detections[idx] for idx in kept]

def main():
    if not HAILO_OK:
        return
        
    print(f"Laden van Classifiers uit {CLASSIFIER_DIR} op CPU...")
    classifier = TwoStageGarbageClassifier(model_dir=CLASSIFIER_DIR, stage1_threshold=0.60, stage2_threshold=0.55)

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": RESOLUTION, "format": "RGB888"})
    picam2.configure(config)

    print(f"Laden van Hailo YOLO model: {HEF_MODEL_PATH}...")
    with Hailo(HEF_MODEL_PATH) as hailo:
        model_h, model_w, _ = hailo.get_input_shape()
        picam2.start()
        print("Camera gestart. Druk op 'Q' om te stoppen.")

        while True:
            t0 = time.time()
            frame = picam2.capture_array()
            h, w = frame.shape[:2]

            # 1. YOLO detectie via Hailo HAT
            resized = cv2.resize(frame, (model_w, model_h))
            hailo_output = hailo.run(resized)
            detections = parse_hailo_output(hailo_output, w, h)

            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 2. Classifier (Stage 1 & 2) via CPU voor elke detectie
            for x1, y1, x2, y2, conf, cls in detections:
                pad = 8
                crop = display[
                    max(0, y1 - pad): min(h, y2 + pad),
                    max(0, x1 - pad): min(w, x2 + pad)
                ]

                # Classificeer de uitsnede
                final_label, clf_conf = classifier.classify(crop)
                
                # Teken box
                main_cat = final_label.split('/')[0] if '/' in final_label else final_label
                color = CATEGORY_COLORS.get(main_cat, (200, 200, 200))
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

                text = f"{final_label} ({clf_conf:.0%})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(display, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(display, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            fps = 1 / (time.time() - t0)
            cv2.putText(display, f"FPS: {fps:.1f} [Hailo + CPU Classifier]", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow("Smart Bin 2-Stage", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

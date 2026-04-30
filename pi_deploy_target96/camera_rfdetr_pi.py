#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import supervision as sv

try:
    from picamera2 import Picamera2
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Picamera2 ontbreekt. Installeer op de Pi via:\n"
        "  sudo apt install python3-libcamera python3-picamera2\n"
        "Start daarna dit script in een omgeving die picamera2 kan importeren.\n"
        "Tip: gebruik system Python of een venv met --system-site-packages."
    ) from exc

try:
    import cv2
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "OpenCV ontbreekt. Installeer via:\n"
        "  python -m pip install opencv-python"
    ) from exc

from rfdetr import RFDETRBase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF-DETR objectdetectie via Pi camera (CPU)"
    )
    parser.add_argument(
        "--model",
        default="model_best_ema_target96.pth",
        help="Pad naar .pth checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Camera breedte",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Camera hoogte",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=1.0,
        help="Camera warmup in seconden",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Toon live venster",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log elke N frames naar console",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop na N frames (0 = oneindig)",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="Map om geannoteerde frames op te slaan (optioneel)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=30,
        help="Sla elke N frames op als save-dir is gezet",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model niet gevonden: {model_path}", file=sys.stderr)
        return 2

    display = bool(args.display)
    if display and not os.environ.get("DISPLAY"):
        print("Geen DISPLAY gevonden; display wordt uitgezet.")
        display = False

    save_dir = Path(args.save_dir).expanduser() if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    model = RFDETRBase(pretrain_weights=str(model_path), device="cpu")
    class_names = model.class_names

    camera = Picamera2()
    cfg = camera.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}
    )
    camera.configure(cfg)
    camera.start()
    time.sleep(max(args.warmup, 0.0))

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5)

    frame_idx = 0
    try:
        while True:
            frame_rgb = camera.capture_array()
            if frame_rgb is None:
                print("Geen frame ontvangen", file=sys.stderr)
                break

            detections = model.predict(frame_rgb, threshold=args.threshold)

            labels = []
            if len(detections) > 0:
                for conf, class_id in zip(detections.confidence, detections.class_id):
                    name = class_names[int(class_id)] if class_id is not None else "onbekend"
                    labels.append(f"{name} {float(conf):.2f}")

            frame_bgr = frame_rgb[:, :, ::-1]
            annotated = box_annotator.annotate(
                scene=frame_bgr.copy(),
                detections=detections,
            )
            annotated = label_annotator.annotate(
                scene=annotated,
                detections=detections,
                labels=labels,
            )

            if frame_idx % max(args.log_every, 1) == 0:
                if len(detections) == 0:
                    print(f"frame {frame_idx}: geen detecties")
                else:
                    best_idx = int(np.argmax(detections.confidence))
                    best_class_id = int(detections.class_id[best_idx])
                    best_label = class_names[best_class_id]
                    best_conf = float(detections.confidence[best_idx])
                    print(
                        f"frame {frame_idx}: {len(detections)} detecties | top: {best_label} {best_conf:.2f}"
                    )

            if save_dir and frame_idx % max(args.save_every, 1) == 0:
                out_path = save_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), annotated)

            if display:
                cv2.imshow("RF-DETR", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
    finally:
        camera.stop()
        if display:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

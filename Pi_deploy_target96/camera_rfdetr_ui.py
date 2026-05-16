#!/usr/bin/env python3
import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv

try:
    import onnxruntime as ort
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "onnxruntime ontbreekt. Installeer via:\n"
        "  python -m pip install onnxruntime"
    ) from exc

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
    import cv2  # noqa: F401
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "OpenCV ontbreekt. Installeer via:\n"
        "  python -m pip install opencv-python"
    ) from exc

try:
    from PIL import Image, ImageTk
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Pillow ontbreekt. Installeer via:\n"
        "  python -m pip install pillow"
    ) from exc

try:
    import tkinter as tk
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Tkinter ontbreekt. Installeer via:\n"
        "  sudo apt install python3-tk"
    ) from exc

from rfdetr import RFDETRBase

DEFAULT_STAGE1_CLASSES = [
    "Organisch",
    "PMD",
    "Papier",
    "Restafval",
    "Overige",
]
DEFAULT_STAGE2_CLASSES = [
    "Batterijen",
    "Elektronica",
    "Glas",
    "Lightbulbs",
    "Metaal",
]


@dataclass
class UiState:
    latest_frame: np.ndarray | None = None
    last_detections: sv.Detections | None = None
    last_labels: list[str] | None = None
    last_class_label: str | None = None
    last_class_conf: float | None = None
    status_text: str = "Starten..."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RF-DETR objectdetectie via Pi camera met UI (CPU)"
    )
    default_ai_dir = Path(__file__).resolve().parents[1] / "Code PI" / "AI"
    parser.add_argument(
        "--model",
        default="model_best_ema_target96.pth",
        help="Pad naar .pth checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Camera breedte",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=240,
        help="Camera hoogte",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=1.0,
        help="Camera warmup in seconden",
    )
    parser.add_argument(
        "--update-ms",
        type=int,
        default=60,
        help="UI refresh interval in ms",
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=1.0,
        help="Window scale relative to camera size (e.g. 1.0, 0.75, 1.5)",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="Run inference every N frames (1 = live)",
    )
    parser.add_argument(
        "--infer-shape",
        default="auto",
        help="Resize input for inference, e.g. 320x320 or auto",
    )
    parser.add_argument(
        "--stage1-model",
        default=str(default_ai_dir / "stage1_main.onnx"),
        help="Pad naar stage1 ONNX model",
    )
    parser.add_argument(
        "--stage2-model",
        default=str(default_ai_dir / "stage2_overige.onnx"),
        help="Pad naar stage2 ONNX model",
    )
    parser.add_argument(
        "--two-stage-metadata",
        default=str(default_ai_dir / "two_stage_metadata.json"),
        help="Pad naar metadata JSON voor two-stage",
    )
    parser.add_argument(
        "--disable-two-stage",
        action="store_true",
        help="Disable two-stage classification",
    )
    return parser.parse_args()


class RfDetrUiApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.running = True
        self.state = UiState()
        self.state_lock = threading.Lock()
        self.infer_every = max(1, int(args.infer_every))
        self.infer_shape = self._parse_infer_shape(args.infer_shape)
        self.window_scale = max(0.5, float(args.window_scale))
        self.last_infer_counter = -1
        self.frame_counter = 0

        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model niet gevonden: {model_path}")

        if not os.environ.get("DISPLAY"):
            raise RuntimeError("Geen DISPLAY gevonden. Start dit script in een grafische sessie.")

        self.model = RFDETRBase(pretrain_weights=str(model_path), device="cpu")
        self.class_names = self.model.class_names

        self.two_stage_enabled = False
        self.stage1_session: ort.InferenceSession | None = None
        self.stage2_session: ort.InferenceSession | None = None
        self.stage1_input_name: str | None = None
        self.stage1_input_shape = None
        self.stage2_input_name: str | None = None
        self.stage2_input_shape = None
        self.stage1_classes = list(DEFAULT_STAGE1_CLASSES)
        self.stage2_overige_classes = list(DEFAULT_STAGE2_CLASSES)
        self.main_label_for_stage2 = "Overige"
        self.default_fallback = "Restafval"
        self.stage1_confidence_threshold = 0.4
        self.stage2_confidence_threshold = 0.45
        self._load_two_stage(args)

        self.camera = Picamera2()
        cfg = self.camera.create_preview_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"}
        )
        self.camera.configure(cfg)
        self.camera.start()
        time.sleep(max(args.warmup, 0.0))

        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5)

        self.root = tk.Tk()
        self.root.title("RF-DETR Objectdetectie")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Escape>", lambda _event: self.on_close())
        self.root.attributes("-fullscreen", False)
        self.root.state("normal")

        window_w = max(320, int(self.args.width * self.window_scale))
        window_h = max(240, int(self.args.height * self.window_scale) + 60)
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        pos_x = max(0, (screen_w - window_w) // 2)
        pos_y = max(0, (screen_h - window_h) // 2)
        self.root.geometry(f"{window_w}x{window_h}+{pos_x}+{pos_y}")
        self.root.resizable(True, True)

        self.image_label = tk.Label(self.root, bg="black")
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Opstarten...")
        self.status_label = tk.Label(self.root, textvariable=self.status_var, font=("Helvetica", 14))
        self.status_label.pack(side=tk.BOTTOM, padx=10, pady=10)

        capture_worker = threading.Thread(target=self._capture_loop, daemon=True)
        infer_worker = threading.Thread(target=self._inference_loop, daemon=True)
        capture_worker.start()
        infer_worker.start()

        self._schedule_ui_update()

    def on_close(self) -> None:
        self.running = False
        self.root.after(100, self.root.destroy)

    @staticmethod
    def _parse_infer_shape(value: str) -> tuple[int, int] | None:
        raw = value.strip()
        if not raw:
            return None
        if raw.lower() in {"auto", "none", "default"}:
            return None
        parts = raw.lower().replace("x", ",").split(",")
        parts = [p for p in parts if p]
        if len(parts) != 2:
            raise ValueError("--infer-shape must be like 320x320")
        height = int(parts[0])
        width = int(parts[1])
        if height <= 0 or width <= 0:
            raise ValueError("--infer-shape must be positive integers")
        return (height, width)

    @staticmethod
    def _parse_threshold(metadata: dict, keys: tuple[str, ...], default: float) -> float:
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, (int, float)):
                return max(0.0, min(1.0, float(value)))
        return default

    def _load_two_stage(self, args: argparse.Namespace) -> None:
        if args.disable_two_stage:
            return

        stage1_path = Path(args.stage1_model).expanduser()
        stage2_path = Path(args.stage2_model).expanduser()
        if not stage1_path.is_file() or not stage2_path.is_file():
            return

        metadata: dict = {}
        metadata_path = Path(args.two_stage_metadata).expanduser()
        if metadata_path.is_file():
            try:
                with open(metadata_path, "r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except Exception:
                metadata = {}

        self.stage1_classes = list(metadata.get("stage1_classes") or self.stage1_classes)
        self.stage2_overige_classes = list(metadata.get("stage2_overige_classes") or self.stage2_overige_classes)
        self.main_label_for_stage2 = str(metadata.get("main_label_for_stage2") or self.main_label_for_stage2)
        self.default_fallback = str(metadata.get("default_fallback") or self.default_fallback)
        self.stage1_confidence_threshold = self._parse_threshold(
            metadata,
            keys=("stage1_confidence_threshold", "stage1_threshold", "stage1_conf_threshold"),
            default=self.stage1_confidence_threshold,
        )
        self.stage2_confidence_threshold = self._parse_threshold(
            metadata,
            keys=("stage2_confidence_threshold", "stage2_threshold", "stage2_conf_threshold"),
            default=self.stage2_confidence_threshold,
        )

        self.stage1_session = ort.InferenceSession(
            str(stage1_path),
            providers=["CPUExecutionProvider"],
        )
        self.stage2_session = ort.InferenceSession(
            str(stage2_path),
            providers=["CPUExecutionProvider"],
        )
        self.stage1_input_name = self.stage1_session.get_inputs()[0].name
        self.stage1_input_shape = self.stage1_session.get_inputs()[0].shape
        self.stage2_input_name = self.stage2_session.get_inputs()[0].name
        self.stage2_input_shape = self.stage2_session.get_inputs()[0].shape
        self.two_stage_enabled = True

    @staticmethod
    def _to_probabilities(raw_output: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw_output, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise RuntimeError("Lege model output")
        if np.all(arr >= 0.0):
            total = float(arr.sum())
            if 0.98 <= total <= 1.02:
                return arr
        exp_x = np.exp(arr - np.max(arr))
        return exp_x / max(float(exp_x.sum()), 1e-9)

    @staticmethod
    def _preprocess_image(image: np.ndarray, input_shape=None) -> np.ndarray:
        target_h, target_w = 224, 224
        if input_shape:
            try:
                shape = input_shape
                if len(shape) == 4:
                    if isinstance(shape[2], int) and isinstance(shape[3], int):
                        if shape[2] > 0 and shape[3] > 0:
                            target_h, target_w = shape[2], shape[3]
                    elif isinstance(shape[1], int) and isinstance(shape[2], int):
                        if shape[1] > 0 and shape[2] > 0 and (shape[3] == 3 or shape[3] == 1):
                            target_h, target_w = shape[1], shape[2]
            except Exception:
                pass

        img = Image.fromarray(image).resize((target_w, target_h))
        img_array = np.array(img).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        return img_array

    def _run_two_stage(self, image: np.ndarray) -> tuple[str, float]:
        if self.stage1_session is None or self.stage1_input_name is None:
            raise RuntimeError("Stage 1 model niet geladen")

        img1 = self._preprocess_image(image, self.stage1_input_shape)
        out1 = self.stage1_session.run(None, {self.stage1_input_name: img1})
        probs1 = self._to_probabilities(np.asarray(out1[0], dtype=np.float32))
        idx1 = int(np.argmax(probs1))

        if idx1 >= len(self.stage1_classes):
            raise RuntimeError("Stage 1 output index buiten bereik")

        label1 = self.stage1_classes[idx1]
        conf1 = float(probs1[idx1])
        final_label = label1
        final_conf = conf1

        if conf1 < self.stage1_confidence_threshold:
            final_label = self.default_fallback
            final_conf = conf1
        elif label1 == self.main_label_for_stage2:
            if self.stage2_session is None or self.stage2_input_name is None:
                raise RuntimeError("Stage 2 model niet geladen")
            img2 = self._preprocess_image(image, self.stage2_input_shape)
            out2 = self.stage2_session.run(None, {self.stage2_input_name: img2})
            probs2 = self._to_probabilities(np.asarray(out2[0], dtype=np.float32))
            idx2 = int(np.argmax(probs2))
            if idx2 >= len(self.stage2_overige_classes):
                raise RuntimeError("Stage 2 output index buiten bereik")

            sub_label = self.stage2_overige_classes[idx2]
            conf2 = float(probs2[idx2])
            if conf2 < self.stage2_confidence_threshold:
                final_label = self.main_label_for_stage2
                final_conf = conf2
            else:
                if sub_label.startswith(f"{self.main_label_for_stage2}/"):
                    final_label = sub_label
                else:
                    final_label = f"{self.main_label_for_stage2}/{sub_label}"
                final_conf = conf2

        return final_label, final_conf

    @staticmethod
    def _crop_from_detection(frame_rgb: np.ndarray, detections: sv.Detections, idx: int) -> np.ndarray | None:
        if not hasattr(detections, "xyxy"):
            return None
        if idx < 0 or idx >= len(detections):
            return None
        xyxy = detections.xyxy[idx]
        if xyxy is None or len(xyxy) != 4:
            return None
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        h, w = frame_rgb.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_rgb[y1:y2, x1:x2]

    def _capture_loop(self) -> None:
        try:
            while self.running:
                frame_bgr = self.camera.capture_array()
                with self.state_lock:
                    self.state.latest_frame = frame_bgr
                    self.frame_counter += 1
        except Exception as exc:
            with self.state_lock:
                self.state.status_text = f"Fout: {exc}"
        finally:
            try:
                self.camera.stop()
            except Exception:
                pass

    def _inference_loop(self) -> None:
        last_status = "Wachten op inferentie..."
        try:
            while self.running:
                with self.state_lock:
                    frame_bgr = self.state.latest_frame
                    frame_counter = self.frame_counter

                if frame_bgr is None:
                    time.sleep(0.01)
                    continue

                if frame_counter == self.last_infer_counter:
                    time.sleep(0.005)
                    continue

                if frame_counter % self.infer_every != 0:
                    time.sleep(0.002)
                    continue

                self.last_infer_counter = frame_counter

                frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
                detections = self.model.predict(
                    frame_rgb,
                    threshold=self.args.threshold,
                    shape=self.infer_shape,
                )

                labels = []
                status_text = "Geen detecties"
                best_idx = None
                best_conf = None
                class_label = None
                class_conf = None

                if len(detections) > 0:
                    best_idx = int(np.argmax(detections.confidence))
                    best_conf = float(detections.confidence[best_idx])

                    if self.two_stage_enabled:
                        crop = self._crop_from_detection(frame_rgb, detections, best_idx)
                        if crop is not None:
                            try:
                                class_label, class_conf = self._run_two_stage(crop)
                            except Exception as exc:
                                class_label = None
                                class_conf = None
                                status_text = f"Two-stage fout: {exc}"

                    labels = [f"{float(conf):.2f}" for conf in detections.confidence]
                    if best_idx is not None and 0 <= best_idx < len(labels):
                        if class_label:
                            labels[best_idx] = f"{class_label} {float(class_conf or 0.0):.2f}"
                        elif best_conf is not None:
                            labels[best_idx] = f"top {best_conf:.2f}"

                    if class_label:
                        status_text = (
                            f"Klassificatie: {class_label} {float(class_conf or 0.0):.2f}"
                            f" | detecties: {len(detections)}"
                        )
                    else:
                        status_text = f"Detecties: {len(detections)} | top conf: {best_conf:.2f}"

                last_status = status_text
                with self.state_lock:
                    self.state.last_detections = detections
                    self.state.last_labels = labels
                    self.state.last_class_label = class_label
                    self.state.last_class_conf = class_conf
                    self.state.status_text = status_text
        except Exception as exc:
            with self.state_lock:
                self.state.status_text = f"Fout: {exc}"

    def _schedule_ui_update(self) -> None:
        if not self.running:
            return

        frame_rgb = None
        last_detections = None
        last_labels = None
        status_text = ""
        with self.state_lock:
            frame_rgb = self.state.latest_frame
            last_detections = self.state.last_detections
            last_labels = self.state.last_labels
            status_text = self.state.status_text

        if frame_rgb is not None:
            frame_bgr = frame_rgb
            display_frame = frame_bgr[:, :, ::-1]
            if last_detections is not None:
                annotated_bgr = self.box_annotator.annotate(
                    scene=frame_bgr.copy(),
                    detections=last_detections,
                )
                annotated_bgr = self.label_annotator.annotate(
                    scene=annotated_bgr,
                    detections=last_detections,
                    labels=last_labels or [],
                )
                display_frame = annotated_bgr[:, :, ::-1]

            image = Image.fromarray(display_frame)
            target_w = self.image_label.winfo_width()
            target_h = self.image_label.winfo_height()
            if target_w <= 1 or target_h <= 1:
                target_w = self.image_label.winfo_reqwidth() or self.args.width
                target_h = self.image_label.winfo_reqheight() or self.args.height
            image = self._resize_to_fit(image, target_w, target_h)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

        if status_text:
            self.status_var.set(status_text)

        self.root.after(self.args.update_ms, self._schedule_ui_update)

    @staticmethod
    def _resize_to_fit(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        if target_w <= 0 or target_h <= 0:
            return image
        img_w, img_h = image.size
        if img_w <= 0 or img_h <= 0:
            return image
        scale = min(target_w / img_w, target_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        if new_w == img_w and new_h == img_h:
            return image
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    args = parse_args()
    app = RfDetrUiApp(args)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

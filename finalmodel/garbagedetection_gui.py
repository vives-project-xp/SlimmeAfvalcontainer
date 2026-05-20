"""
Slimme Afvalcontainer – GUI with RF-DETR object detection.

Inference priority:
  1. Hailo detector + two-stage ONNX classifier in this folder
  2. YOLO ONNX detector + two-stage ONNX classifier in this folder
  3. Hailo HEF  (.hef)   – legacy classification path via hailo_platform
  4. RF-DETR    (.pth)   – PyTorch CPU/GPU object detection
  5. Two-stage ONNX      – stage1_main.onnx + stage2_overige.onnx
  6. Single-stage ONNX   – model.onnx / inference_model.onnx
"""

from __future__ import annotations

import argparse
import cv2
import json
import logging
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import tkinter as tk
from tkinter import font as tkfont

# ── Optional heavy imports ──────────────────────────────────────────────────

try:
    import onnxruntime as ort
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

try:
    from picamera2 import Picamera2
    _PICAM_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _PICAM_AVAILABLE = False

try:
    from picamera2.devices.hailo import Hailo as PiCameraHailo  # type: ignore
    _PICAM_HAILO_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    _PICAM_HAILO_AVAILABLE = False

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageOps
except ImportError as exc:
    raise ImportError(
        "PIL.ImageTk ontbreekt.\n  sudo apt install python3-pil.imagetk python3-tk"
    ) from exc

try:
    from led_controller import LedController  # type: ignore
    _LED_AVAILABLE = True
except ImportError:
    _LED_AVAILABLE = False

try:
    from hailo_platform import (  # type: ignore
        VDevice, HEF, ConfigureParams, HailoStreamInterface,
        InferVStreams, InputVStreamParams, OutputVStreamParams,
    )
    _HAILO_AVAILABLE = True
except ImportError:
    _HAILO_AVAILABLE = False

try:
    from rfdetr import RFDETRBase  # type: ignore
    _RFDETR_AVAILABLE = True
except ImportError:
    _RFDETR_AVAILABLE = False

try:
    from detector import YOLODetector  # type: ignore
    _YOLO_DETECTOR_AVAILABLE = True
except ImportError:
    _YOLO_DETECTOR_AVAILABLE = False

try:
    from classifier import TwoStageGarbageClassifier  # type: ignore
    _LOCAL_TWO_STAGE_AVAILABLE = True
except ImportError:
    _LOCAL_TWO_STAGE_AVAILABLE = False

from ultrasone_controller import UltrasonicMonitor

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_CLASSES = (
    "Organisch", "Overige/Batterijen", "Overige/Elektronica",
    "Overige/Glas", "Overige/Lightbulbs", "Overige/Metaal",
    "PMD", "Papier", "Restafval",
)

CLASS_COLOR_MAP: dict[str, str] = {
    "Organisch":          "#4CAF50",
    "PMD":                "#FFC107",
    "Papier":             "#2196F3",
    "Restafval":          "#757575",
    "Overige":            "#9E9E9E",
    "Overige/Batterijen": "#FF7043",
    "Overige/Elektronica":"#EF5350",
    "Overige/Glas":       "#26A69A",
    "Overige/Lightbulbs": "#FFD54F",
    "Overige/Metaal":     "#8D6E63",
}

C_BG      = "#F5F5F0"
C_HEADER  = "#FFFFFF"
C_BORDER  = "#CCCCCC"
C_TEXT    = "#222222"
C_SUBTEXT = "#666666"
C_ACCENT  = "#3498DB"
C_SUCCESS = "#2ECC71"
C_ERROR   = "#E74C3C"

HAND_SKIN_RATIO_STRONG = 0.55
HAND_SKIN_RATIO_EDGE = 0.35
HAND_SKIN_RATIO_GENERAL = 0.45
HAND_LARGE_AREA_RATIO = 0.10
HAND_EDGE_MARGIN_PX = 12
HAND_EDGE_AREA_RATIO = 0.03
HAND_SKIN_RATIO_LOOSE_EDGE = 0.18
MAX_CLASSIFIED_DETECTIONS = 3
MAX_OVERLAY_BOXES = 1
MIN_DETECTION_CONFIDENCE = 0.40
MIN_COMBINED_SCORE = 0.12
MIN_RESTAFVAL_CONFIDENCE = 0.68

BIN_DEFS = [
    {"key": "org",    "label": "Organisch", "idle": "#C8E6C9", "active": "#4CAF50", "border": "#388E3C"},
    {"key": "papier", "label": "Papier",    "idle": "#BBDEFB", "active": "#2196F3", "border": "#1565C0"},
    {"key": "pmd",    "label": "PMD",       "idle": "#FFF9C4", "active": "#FFC107", "border": "#F57F17"},
    {"key": "rest",   "label": "Restafval", "idle": "#E0E0E0", "active": "#757575", "border": "#424242"},
]

# Keywords that map a detected class name to a bin
_BIN_KEYWORDS: dict[str, list[str]] = {
    "org":    ["organisch", "organic", "gft", "bio", "food", "groente", "fruit", "etens"],
    "papier": ["papier", "paper", "karton", "cardboard", "krant", "boek", "tijdschrift"],
    "pmd":    ["pmd", "plastic", "metaal", "metal", "blik", "fles", "fles", "drank",
               "verpakking", "blikje", "flesje", "drankfles"],
}


# ── Config dataclasses ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class DisplayConfig:
    model_path: str | None = None
    window_width: int  = 1024
    window_height: int = 600
    fullscreen: bool   = False
    rotate: int        = 0
    update_ms: int     = 50
    det_threshold: float = 0.45
    ignore_hands: bool = True
    detection_logging: bool = True
    ultra_debug: bool = False
    ultra_debug_every_cycles: int = 1
    org_ultra_trig: int | None = None
    org_ultra_echo: int | None = None
    pmd_ultra_trig: int | None = None
    pmd_ultra_echo: int | None = None


@dataclass(frozen=True)
class TwoStageConfig:
    stage1_path: str
    stage2_path: str
    metadata_path: str | None = None


# ── Hailo engine ───────────────────────────────────────────────────────────

class HailoEngine:
    def __init__(self, hef_path: str):
        if not _HAILO_AVAILABLE:
            raise RuntimeError("hailo_platform niet beschikbaar")
        self._target = VDevice()
        self._hef    = HEF(hef_path)
        params       = ConfigureParams.create_from_hef(hef=self._hef, interface=HailoStreamInterface.PCIe)
        self._ng     = self._target.configure(self._hef, params)[0]
        self._ng_p   = self._ng.create_params()
        self._in_p   = InputVStreamParams.make(self._ng)
        self._out_p  = OutputVStreamParams.make(self._ng)
        in_info       = self._hef.get_input_vstream_infos()[0]
        self.input_name  = in_info.name
        self.input_shape = tuple(in_info.shape)   # (H, W, C)

    def infer(self, image_nhwc: np.ndarray) -> np.ndarray:
        with InferVStreams(self._ng, self._in_p, self._out_p) as pipeline:
            with self._ng.activate(self._ng_p):
                results = pipeline.infer({self.input_name: image_nhwc})
        return np.asarray(next(iter(results.values())), dtype=np.float32).reshape(-1)

    def close(self):
        try:
            del self._ng
            del self._target
        except Exception:
            pass


class PiCameraHailoDetector:
    """Small wrapper around picamera2's Hailo helper for a detector HEF."""

    def __init__(self, hef_path: str):
        if not _PICAM_HAILO_AVAILABLE:
            raise RuntimeError("picamera2 Hailo helper niet beschikbaar")
        self._device = PiCameraHailo(hef_path)
        if hasattr(self._device, "__enter__"):
            entered = self._device.__enter__()
            if entered is not None:
                self._device = entered
        self.input_shape = tuple(self._device.get_input_shape())

    def run(self, frame_rgb: np.ndarray):
        return self._device.run(frame_rgb)

    def close(self) -> None:
        if hasattr(self._device, "__exit__"):
            try:
                self._device.__exit__(None, None, None)
            except Exception:
                pass


# ── RF-DETR engine ─────────────────────────────────────────────────────────

class RFDETREngine:
    """
    Wraps RFDETRBase for single-frame waste detection.
    Returns (label, confidence, bbox_xyxy_normalized | None).
    """

    def __init__(self, model_path: str, threshold: float = 0.45):
        if not _RFDETR_AVAILABLE:
            raise RuntimeError("rfdetr pakket niet beschikbaar. Installeer: pip install rfdetr")
        print(f"RF-DETR laden: {model_path}  (dit kan even duren op Pi CPU)")
        self.threshold = threshold
        self.model     = RFDETRBase(pretrain_weights=str(model_path), device="cpu")
        raw_names = getattr(self.model, "class_names", {})
        self.class_names = self._normalize_class_names(raw_names)
        print(f"RF-DETR geladen. Klassen: {self.class_names}")

    @staticmethod
    def _normalize_class_names(raw_names) -> dict[int, str]:
        if isinstance(raw_names, dict):
            out: dict[int, str] = {}
            for k, v in raw_names.items():
                try:
                    out[int(k)] = str(v)
                except Exception:
                    continue
            return out
        if isinstance(raw_names, (list, tuple)):
            return {i: str(v) for i, v in enumerate(raw_names)}
        return {}

    def detect(self, image: "Image.Image") -> tuple[str | None, float, tuple | None]:
        """
        Run detection on a PIL image.
        Returns (class_name, confidence, (x1,y1,x2,y2) normalised) or (None, 0, None).
        """
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            image.save(tmp_path, format="JPEG", quality=90)
            detections = self.model.predict(tmp_path, threshold=self.threshold)
        finally:
            os.unlink(tmp_path)

        if len(detections) == 0:
            return None, 0.0, None

        # Pick detection with highest confidence
        best_i    = int(np.argmax(detections.confidence))
        conf      = float(detections.confidence[best_i])
        class_id  = int(detections.class_id[best_i])
        class_name= self.class_names.get(class_id, f"klasse_{class_id}")

        # Bounding box in pixel coords → normalise to 0-1
        w, h = image.size
        if detections.xyxy is not None and len(detections.xyxy) > best_i:
            x1, y1, x2, y2 = detections.xyxy[best_i]
            bbox = (float(x1)/w, float(y1)/h, float(x2)/w, float(y2)/h)
        else:
            bbox = None

        return class_name, conf, bbox

    def all_detections(self, image: "Image.Image") -> list[tuple[str, float, tuple | None]]:
        """Return all detections above threshold as list of (name, conf, bbox_norm)."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            image.save(tmp_path, format="JPEG", quality=90)
            detections = self.model.predict(tmp_path, threshold=self.threshold)
        finally:
            os.unlink(tmp_path)

        w, h = image.size
        results = []
        for i in range(len(detections)):
            conf      = float(detections.confidence[i])
            class_id  = int(detections.class_id[i])
            name      = self.class_names.get(class_id, f"klasse_{class_id}")
            if detections.xyxy is not None and len(detections.xyxy) > i:
                x1, y1, x2, y2 = detections.xyxy[i]
                bbox = (float(x1)/w, float(y1)/h, float(x2)/w, float(y2)/h)
            else:
                bbox = None
            results.append((name, conf, bbox))
        return results


# ── Model-path helpers ─────────────────────────────────────────────────────

def _find_hef(model_path: str | None) -> str | None:
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if model_path:
        p = Path(model_path).expanduser()
        if p.suffix == ".hef":
            candidates.append(p if p.is_absolute() else (script_dir / p).resolve())
    for d in [script_dir, script_dir / "AI", script_dir.parent / "Ai-model"]:
        if d.exists():
            candidates += sorted(d.glob("*.hef"))
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _find_detector_hef(model_path: str | None = None) -> str | None:
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if model_path:
        p = Path(model_path).expanduser()
        if p.suffix == ".hef":
            candidates.append(p if p.is_absolute() else (script_dir / p).resolve())
        elif p.is_dir():
            candidates.append((p if p.is_absolute() else (script_dir / p).resolve()) / "yolov8_detector.hef")
    candidates.append((script_dir / "yolov8_detector.hef").resolve())
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _find_detector_onnx(model_path: str | None = None) -> str | None:
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if model_path:
        p = Path(model_path).expanduser()
        if p.suffix == ".onnx" and p.name == "yolov8_detector.onnx":
            candidates.append(p if p.is_absolute() else (script_dir / p).resolve())
        elif p.is_dir():
            candidates.append((p if p.is_absolute() else (script_dir / p).resolve()) / "yolov8_detector.onnx")
    candidates.append((script_dir / "yolov8_detector.onnx").resolve())
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _find_rfdetr(model_path: str | None) -> str | None:
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if model_path:
        p = Path(model_path).expanduser()
        if p.suffix == ".pth":
            candidates.append(p if p.is_absolute() else (script_dir / p).resolve())
    for d in [script_dir, script_dir / "AI", script_dir.parent / "Ai-model",
              script_dir.parent / "pi_deploy_target96"]:
        if d.exists():
            candidates += sorted(d.glob("*.pth"))
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def resolve_model_path(model_path: str | None = None) -> list[str]:
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []
    if model_path:
        p = Path(model_path).expanduser()
        candidates.append(p if p.is_absolute() else (script_dir / p).resolve())
    for stem in ("model", "inference_model", "inference_model.sim"):
        candidates.append((script_dir / f"{stem}.onnx").resolve())
    for sub in ["AI", "../Ai-model"]:
        d = (script_dir / sub).resolve()
        if d.exists():
            for stem in ("model", "inference_model"):
                candidates.append((d / f"{stem}.onnx").resolve())
    seen: set[str] = set()
    valid: list[str] = []
    for c in candidates:
        k = str(c)
        if k not in seen:
            seen.add(k)
            if c.is_file():
                valid.append(k)
    if not valid:
        raise FileNotFoundError("Geen ONNX-model gevonden.")
    return valid


def resolve_two_stage_paths(model_path: str | None = None) -> TwoStageConfig | None:
    script_dir = Path(__file__).resolve().parent
    search_dirs: list[Path] = []
    if model_path:
        p = Path(model_path).expanduser()
        if p.is_dir():
            search_dirs.append(p.resolve() if p.is_absolute() else (script_dir / p).resolve())
    search_dirs += [(script_dir / "AI").resolve(), script_dir.resolve(),
                    (script_dir.parent / "Ai-model").resolve()]
    seen: set[str] = set()
    for root in search_dirs:
        k = str(root)
        if k in seen:
            continue
        seen.add(k)
        s1, s2 = root / "stage1_main.onnx", root / "stage2_overige.onnx"
        if s1.is_file() and s2.is_file():
            meta = root / "two_stage_metadata.json"
            return TwoStageConfig(str(s1), str(s2), str(meta) if meta.is_file() else None)
    return None


def resolve_local_two_stage_dir(model_path: str | None = None) -> str | None:
    two_stage = resolve_two_stage_paths(model_path)
    if not two_stage:
        return None
    return str(Path(two_stage.stage1_path).resolve().parent)


# ── Label → bin mapping ────────────────────────────────────────────────────

def label_to_bin(label: str) -> str:
    """Map any detected class name to a bin key: org / papier / pmd / rest."""
    lower = label.strip().lower()
    if lower == "overige" or lower.startswith("overige/"):
        return ""
    for bin_key, keywords in _BIN_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return bin_key
    return "rest"


def bin_to_led_cmd(bin_key: str) -> str:
    if not bin_key:
        return "reject_blink"
    return {
        "org":    "select_organisch",
        "papier": "select_karton",
        "pmd":    "select_pmd",
        "rest":   "select_rest",
    }.get(bin_key, "idle")


def bin_label(bin_key: str) -> str:
    return {
        "org":    "Organisch",
        "papier": "Papier",
        "pmd":    "PMD",
        "rest":   "Restafval",
    }.get(bin_key, bin_key.upper())


# ── Main GUI ───────────────────────────────────────────────────────────────

class InferenceGUI:
    def __init__(self, config: DisplayConfig):
        self.config   = config
        self.classes  = list(DEFAULT_CLASSES)
        self.running  = True
        self.initialized         = False
        self.worker_active       = False
        self.init_in_progress    = False
        self.auto_init_retry_count   = 0
        self.max_auto_init_retries   = 6
        self.init_retry_delay_ms     = 10_000

        self.result_queue: Queue[tuple[str, object]] = Queue()
        self.latest_frame: np.ndarray | None = None
        self._last_bboxes: list[tuple[str, float, tuple | None]] = []
        self.auto_detect_enabled = True
        self.auto_detect_interval_s = 1.5
        self._next_auto_detect_at = 0.0

        # Engine state
        self.pipeline_mode = "single"
        self.hailo: HailoEngine | None   = None
        self.hailo_detector: PiCameraHailoDetector | None = None
        self.rfdetr: RFDETREngine | None = None
        self.yolo_detector = None
        self.two_stage_classifier = None

        # ONNX state
        self.session           = None
        self.input_name: str | None = None
        self.input_shape       = None
        self.output_names: list[str] = []
        self.stage1_session    = None
        self.stage1_input_name: str | None = None
        self.stage1_input_shape = None
        self.stage2_session    = None
        self.stage2_input_name: str | None = None
        self.stage2_input_shape = None
        self.stage1_classes: list[str]         = ["Organisch", "PMD", "Papier", "Restafval", "Overige"]
        self.stage2_overige_classes: list[str] = ["Batterijen", "Elektronica", "Glas", "Lightbulbs", "Metaal"]
        self.main_label_for_stage2  = "Overige"
        self.default_fallback       = "Restafval"
        self.stage1_confidence_threshold = 0.40
        self.stage2_confidence_threshold = 0.45

        self.camera = None
        self.led    = None
        sensor_overrides: dict[str, dict[str, int] | None] = {}
        if config.org_ultra_trig is not None and config.org_ultra_echo is not None:
            sensor_overrides["org"] = {"trig": config.org_ultra_trig, "echo": config.org_ultra_echo}
        if config.pmd_ultra_trig is not None and config.pmd_ultra_echo is not None:
            sensor_overrides["pmd"] = {"trig": config.pmd_ultra_trig, "echo": config.pmd_ultra_echo}

        self.ultra  = UltrasonicMonitor(
            debug=config.ultra_debug,
            debug_every_cycles=config.ultra_debug_every_cycles,
            sensor_overrides=sensor_overrides,
        )

        self._active_bin: str | None = None
        self._awaiting_throw_bin: str | None = None
        self._throw_timeout_id: str | None = None
        self._bin_frames: dict[str, tk.Frame] = {}
        self._bin_labels: dict[str, tk.Label] = {}
        self._bin_fill_labels: dict[str, tk.Label] = {}
        self.detect_log = self._setup_detection_logger()

        self.setup_ui()
        self.update_ui_state(enabled=False)
        self.set_status("Systeem opstarten...", C_ACCENT)
        self.update_preview()
        self.start_initialization()
        self.root.after(250, self._auto_detect_tick)
        self.root.after(500, self._update_ultrasonic_ui)

    @staticmethod
    def _setup_detection_logger() -> logging.Logger:
        log_path = Path(__file__).resolve().parent / "detect_debug.log"
        logger = logging.getLogger(f"smartbin.detect.{log_path}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
            logger.addHandler(handler)
        return logger

    def _log_detection(self, message: str) -> None:
        if self.config.detection_logging:
            self.detect_log.info(message)

    # ── Initialization ────────────────────────────────────────────────────

    def start_initialization(self) -> None:
        if not self.running or self.init_in_progress or self.initialized:
            return
        self.init_in_progress = True
        threading.Thread(target=self._initialize_worker, daemon=True).start()

    def _initialize_worker(self) -> None:
        try:
            runtime_payload: dict[str, object] = {}

            # 1) Lokale test10mei Light flow: Hailo detector + two-stage ONNX classifier
            local_two_stage_dir = resolve_local_two_stage_dir(self.config.model_path)
            hef_path = _find_detector_hef(self.config.model_path)
            if (
                not runtime_payload
                and local_two_stage_dir
                and hef_path
                and _LOCAL_TWO_STAGE_AVAILABLE
                and _PICAM_HAILO_AVAILABLE
            ):
                try:
                    print(f"[Init] Hailo detector + two-stage classifier: {hef_path}")
                    meta = self._load_metadata(str(Path(local_two_stage_dir) / "two_stage_metadata.json"))
                    s1_cls = list(meta.get("stage1_classes") or self.stage1_classes)
                    s2_cls = list(meta.get("stage2_overige_classes") or self.stage2_overige_classes)
                    mlabel = str(meta.get("main_label_for_stage2") or self.main_label_for_stage2)
                    fallback = str(meta.get("default_fallback") or self.default_fallback)
                    s1_th = self._parse_threshold(meta, ("stage1_confidence_threshold",), 0.60)
                    s2_th = self._parse_threshold(meta, ("stage2_confidence_threshold",), 0.55)
                    hailo_detector = PiCameraHailoDetector(hef_path)
                    classifier = TwoStageGarbageClassifier(
                        model_dir=local_two_stage_dir,
                        stage1_threshold=s1_th,
                        stage2_threshold=s2_th,
                    )
                    runtime_payload = {
                        "pipeline_mode": "hailo_detector_two_stage",
                        "hailo_detector": hailo_detector,
                        "two_stage_classifier": classifier,
                        "stage1_classes": s1_cls,
                        "stage2_overige_classes": s2_cls,
                        "main_label_for_stage2": mlabel,
                        "default_fallback": fallback,
                        "stage1_confidence_threshold": s1_th,
                        "stage2_confidence_threshold": s2_th,
                        "classes": self._build_two_stage_classes(s1_cls, s2_cls, mlabel),
                        "resolved_model": f"{Path(hef_path).name} + two-stage classifier",
                        "output_names": [],
                    }
                except Exception as exc:
                    print(f"[Init] Lokale Hailo detectorflow mislukt: {exc} → fallback")

            # 2) Lokale test10mei Light flow: YOLO ONNX detector + two-stage ONNX classifier
            detector_onnx = _find_detector_onnx(self.config.model_path)
            if (
                not runtime_payload
                and local_two_stage_dir
                and detector_onnx
                and _LOCAL_TWO_STAGE_AVAILABLE
                and _YOLO_DETECTOR_AVAILABLE
            ):
                try:
                    print(f"[Init] YOLO detector + two-stage classifier: {detector_onnx}")
                    meta = self._load_metadata(str(Path(local_two_stage_dir) / "two_stage_metadata.json"))
                    s1_cls = list(meta.get("stage1_classes") or self.stage1_classes)
                    s2_cls = list(meta.get("stage2_overige_classes") or self.stage2_overige_classes)
                    mlabel = str(meta.get("main_label_for_stage2") or self.main_label_for_stage2)
                    fallback = str(meta.get("default_fallback") or self.default_fallback)
                    s1_th = self._parse_threshold(meta, ("stage1_confidence_threshold",), 0.60)
                    s2_th = self._parse_threshold(meta, ("stage2_confidence_threshold",), 0.55)
                    yolo_detector = YOLODetector(
                        detector_onnx,
                        conf_threshold=self.config.det_threshold,
                        iou_threshold=0.35,
                        min_box_area_ratio=0.002,
                        max_box_area_ratio=0.85,
                        max_aspect_ratio=8.0,
                        max_detections=3,
                    )
                    classifier = TwoStageGarbageClassifier(
                        model_dir=local_two_stage_dir,
                        stage1_threshold=s1_th,
                        stage2_threshold=s2_th,
                    )
                    runtime_payload = {
                        "pipeline_mode": "detector_two_stage",
                        "yolo_detector": yolo_detector,
                        "two_stage_classifier": classifier,
                        "stage1_classes": s1_cls,
                        "stage2_overige_classes": s2_cls,
                        "main_label_for_stage2": mlabel,
                        "default_fallback": fallback,
                        "stage1_confidence_threshold": s1_th,
                        "stage2_confidence_threshold": s2_th,
                        "classes": self._build_two_stage_classes(s1_cls, s2_cls, mlabel),
                        "resolved_model": f"{Path(detector_onnx).name} + two-stage classifier",
                        "output_names": [],
                    }
                except Exception as exc:
                    print(f"[Init] Lokale ONNX detectorflow mislukt: {exc} → fallback")

            # 3) Legacy Hailo HEF classification flow
            if not runtime_payload and _HAILO_AVAILABLE:
                hef_path = _find_hef(self.config.model_path)
                if hef_path:
                    try:
                        print(f"[Init] Hailo HEF: {hef_path}")
                        hailo = HailoEngine(hef_path)
                        runtime_payload = {
                            "pipeline_mode": "hailo",
                            "hailo": hailo,
                            "classes": list(DEFAULT_CLASSES),
                            "resolved_model": Path(hef_path).name,
                            "output_names": [],
                        }
                    except Exception as exc:
                        print(f"[Init] Hailo mislukt: {exc} → fallback")

            # 4) RF-DETR .pth
            if not runtime_payload and _RFDETR_AVAILABLE:
                pth_path = _find_rfdetr(self.config.model_path)
                if pth_path:
                    try:
                        print(f"[Init] RF-DETR: {pth_path}")
                        rfdetr = RFDETREngine(pth_path, threshold=self.config.det_threshold)
                        runtime_payload = {
                            "pipeline_mode": "rfdetr",
                            "rfdetr": rfdetr,
                            "classes": list(DEFAULT_CLASSES),
                            "resolved_model": Path(pth_path).name,
                            "output_names": [],
                        }
                    except Exception as exc:
                        print(f"[Init] RF-DETR mislukt: {exc} → fallback")

            # 5) Two-stage ONNX
            if not runtime_payload and _ONNX_AVAILABLE:
                two_stage = resolve_two_stage_paths(self.config.model_path)
                if two_stage:
                    try:
                        s1 = ort.InferenceSession(two_stage.stage1_path)
                        s2 = ort.InferenceSession(two_stage.stage2_path)
                        s1m = s1.get_inputs()[0]; s2m = s2.get_inputs()[0]
                        meta     = self._load_metadata(two_stage.metadata_path)
                        s1_cls   = list(meta.get("stage1_classes") or self.stage1_classes)
                        s2_cls   = list(meta.get("stage2_overige_classes") or self.stage2_overige_classes)
                        mlabel   = str(meta.get("main_label_for_stage2") or self.main_label_for_stage2)
                        fallback = str(meta.get("default_fallback") or self.default_fallback)
                        s1_th    = self._parse_threshold(meta, ("stage1_confidence_threshold",), 0.40)
                        s2_th    = self._parse_threshold(meta, ("stage2_confidence_threshold",), 0.45)
                        combined = self._build_two_stage_classes(s1_cls, s2_cls, mlabel)
                        runtime_payload = {
                            "pipeline_mode": "two_stage",
                            "stage1_session": s1, "stage1_input_name": s1m.name,
                            "stage1_input_shape": s1m.shape,
                            "stage2_session": s2, "stage2_input_name": s2m.name,
                            "stage2_input_shape": s2m.shape,
                            "stage1_classes": s1_cls, "stage2_overige_classes": s2_cls,
                            "main_label_for_stage2": mlabel, "default_fallback": fallback,
                            "stage1_confidence_threshold": s1_th,
                            "stage2_confidence_threshold": s2_th,
                            "classes": combined,
                            "resolved_model": f"{Path(two_stage.stage1_path).name} + {Path(two_stage.stage2_path).name}",
                            "output_names": [o.name for o in s1.get_outputs()],
                        }
                    except Exception as exc:
                        print(f"[Init] Two-stage mislukt: {exc}")

            # 6) Single ONNX
            if not runtime_payload and _ONNX_AVAILABLE:
                for mp in resolve_model_path(self.config.model_path):
                    try:
                        s   = ort.InferenceSession(mp)
                        inm = s.get_inputs()[0]
                        runtime_payload = {
                            "pipeline_mode": "single",
                            "session": s,
                            "input_name": inm.name, "input_shape": inm.shape,
                            "output_names": [o.name for o in s.get_outputs()],
                            "classes": list(DEFAULT_CLASSES),
                            "resolved_model": str(mp),
                        }
                        break
                    except Exception as exc:
                        print(f"[Init] ONNX {mp} mislukt: {exc}")

            if not runtime_payload:
                raise RuntimeError("Geen bruikbaar model gevonden (HEF, PTH of ONNX).")

            print("[Init] Camera initialiseren...")
            if _PICAM_AVAILABLE:
                camera = self._initialize_camera_with_retry()
            else:
                print("[Init] Picamera2 niet beschikbaar – camerapreview uitgeschakeld")
                camera = None

            led = None
            if _LED_AVAILABLE:
                try:
                    led = LedController()
                except Exception as exc:
                    print(f"[Init] LED mislukt: {exc}")

            if not self.running:
                if camera:
                    camera.stop()
                if led:
                    led.close()
                return

            runtime_payload["camera"] = camera
            runtime_payload["led"]    = led
            self.result_queue.put(("init_ok", runtime_payload))

        except Exception as exc:
            self.result_queue.put(("init_error", f"{exc}\n{traceback.format_exc()}"))
        finally:
            self.result_queue.put(("init_finished", None))

    # ── Camera ────────────────────────────────────────────────────────────

    def _initialize_camera_with_retry(self, attempts: int = 5, base_wait_s: float = 1.5):
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            camera = None
            try:
                camera = Picamera2()
                cfg    = camera.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                camera.configure(cfg)
                camera.start()
                time.sleep(1)
                print(f"[Init] Camera OK (poging {attempt})")
                return camera
            except Exception as exc:
                last_error = exc
                if camera:
                    try:
                        camera.stop()
                    except Exception:
                        pass
                if attempt < attempts:
                    time.sleep(base_wait_s * attempt)
        raise RuntimeError(f"Camera mislukt na {attempts} pogingen: {last_error}")

    def update_preview(self) -> None:
        if not self.running:
            return
        if self.camera is None:
            self.root.after(self.config.update_ms, self.update_preview)
            return
        try:
            image = self.camera.capture_array()
        except Exception as exc:
            self.set_status(f"Camera fout: {exc}", C_ERROR)
            self.root.after(self.config.update_ms, self.update_preview)
            return

        image = image[:, :, ::-1]   # BGR→RGB
        self.latest_frame = image.copy()

        img = Image.fromarray(image)
        if self.config.rotate:
            img = img.rotate(self.config.rotate, expand=True)
        if self._last_bboxes:
            img = self._draw_bboxes(img, self._last_bboxes)

        img = self._fit_to_preview(img)

        photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=photo, text="")
        self.preview_label.image = photo

        self.root.after(self.config.update_ms, self.update_preview)

    # ── UI setup ──────────────────────────────────────────────────────────

    def setup_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Slimme Vuilbak")
        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")
        self.root.configure(bg=C_BG)
        self.root.resizable(True, True)

        if self.config.fullscreen:
            self.root.attributes("-fullscreen", True)

        self.root.bind("<Escape>", lambda _e: self.toggle_fullscreen())
        self.root.bind("<F11>",    lambda _e: self.toggle_fullscreen())
        self.root.bind("<space>",  lambda _e: self.classify_threaded(analyze_full_frame=True))
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # ── Header ──────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=C_HEADER,
                          highlightbackground=C_BORDER, highlightthickness=1)
        header.pack(fill=tk.X, padx=16, pady=(12, 6))

        tk.Label(
            header, text="Slimme Vuilbak",
            font=tkfont.Font(family="Helvetica", size=22, weight="bold"),
            bg=C_HEADER, fg=C_TEXT, pady=10
        ).pack(side=tk.LEFT, padx=20)

        self.status_var = tk.StringVar(value="Opstarten...")
        self.status_lbl = tk.Label(
            header, textvariable=self.status_var,
            font=("Helvetica", 10), bg=C_HEADER, fg=C_SUBTEXT
        )
        self.status_lbl.pack(side=tk.RIGHT, padx=20)

        # ── Camera area ─────────────────────────────────────────────────
        cam_outer = tk.Frame(self.root, bg=C_BG)
        cam_outer.pack(fill=tk.X, expand=False, padx=16, pady=4)
        cam_outer.configure(height=max(260, int(self.config.window_height * 0.52)))
        cam_outer.pack_propagate(False)

        cam_border = tk.Frame(cam_outer, bg=C_BORDER)
        cam_border.pack(fill=tk.BOTH, expand=True)

        self.preview_label = tk.Label(
            cam_border, bg="#111111",
            text="Camera laden...",
            font=("Helvetica", 16), fg="#888888"
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        # Overlays drawn on top of camera
        self.prediction_var = tk.StringVar(value="")
        self.prediction_overlay = tk.Label(
            cam_border, textvariable=self.prediction_var,
            font=("Helvetica", 26, "bold"),
            bg="#000000", fg="white", padx=12, pady=6
        )
        self.confidence_var = tk.StringVar(value="")
        self.confidence_overlay = tk.Label(
            cam_border, textvariable=self.confidence_var,
            font=("Helvetica", 13),
            bg="#000000", fg="#CCCCCC", padx=10, pady=4
        )

        # ── Bottom bar ──────────────────────────────────────────────────
        bottom = tk.Frame(self.root, bg=C_BG)
        bottom.pack(fill=tk.X, padx=16, pady=(6, 14))

        btn_col = tk.Frame(bottom, bg=C_BG)
        btn_col.pack(side=tk.LEFT, padx=(0, 20))

        self.btn_classify = tk.Button(
            btn_col, text="ANALYSEER",
            command=lambda: self.classify_threaded(analyze_full_frame=True),
            font=("Helvetica", 13, "bold"),
            bg=C_ACCENT, fg="white",
            activebackground="#2980B9", activeforeground="white",
            bd=0, padx=14, pady=20, cursor="hand2", relief=tk.FLAT
        )
        self.btn_classify.pack(fill=tk.Y, expand=True)

        self.btn_reset = tk.Button(
            btn_col, text="Reset",
            command=self.reset_classification,
            font=("Helvetica", 10),
            bg="#CCCCCC", fg=C_TEXT,
            activebackground="#BBBBBB",
            bd=0, padx=10, pady=6, cursor="hand2", relief=tk.FLAT
        )
        self.btn_reset.pack(fill=tk.X, pady=(6, 0))

        bins_frame = tk.Frame(bottom, bg=C_BG)
        bins_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for i in range(len(BIN_DEFS)):
            bins_frame.columnconfigure(i, weight=1, uniform="bin")
        bins_frame.rowconfigure(0, weight=1)

        for _i, b in enumerate(BIN_DEFS):
            col = tk.Frame(bins_frame, bg=C_BG)
            col.grid(row=0, column=_i, sticky="nsew", padx=8)

            tk.Label(
                col, text=b["label"],
                font=("Helvetica", 10, "bold"),
                bg=C_BG, fg=C_SUBTEXT
            ).pack()

            bin_frame = tk.Frame(
                col, bg=b["idle"],
                highlightbackground=b["border"], highlightthickness=3, bd=0
            )
            bin_frame.pack(fill=tk.X, expand=False)
            bin_frame.configure(height=110)
            bin_frame.pack_propagate(False)

            icon = tk.Label(
                bin_frame, text="♻",
                font=("Helvetica", 24),
                bg=b["idle"], fg=b["border"]
            )
            icon.pack(expand=True)

            fill_lbl = tk.Label(
                col, text="",
                font=("Helvetica", 9),
                bg=C_BG, fg=C_SUBTEXT
            )
            fill_lbl.pack(pady=(4, 0))

            self._bin_frames[b["key"]] = bin_frame
            self._bin_labels[b["key"]] = icon
            self._bin_fill_labels[b["key"]] = fill_lbl

        self.root.after(50, self._process_worker_messages)

    # ── Inference ─────────────────────────────────────────────────────────

    def classify_threaded(self, analyze_full_frame: bool = False) -> None:
        if not self.running or self.worker_active or not self.initialized:
            return
        self.worker_active = True
        self.update_ui_state(enabled=False)
        self.set_status("Analyseren...", C_ACCENT)
        self.prediction_var.set("Bezig...")
        self.confidence_var.set("")
        threading.Thread(
            target=self._classify_worker,
            kwargs={"analyze_full_frame": analyze_full_frame},
            daemon=True,
        ).start()

    def _auto_detect_tick(self) -> None:
        if not self.running:
            return
        now = time.time()
        if (
            self.auto_detect_enabled
            and self.initialized
            and not self.worker_active
            and self._awaiting_throw_bin is None
            and now >= self._next_auto_detect_at
        ):
            self._next_auto_detect_at = now + self.auto_detect_interval_s
            if self._should_analyze_current_frame():
                self.classify_threaded()
            else:
                self._show_no_detection_idle_state()
        self.root.after(200, self._auto_detect_tick)

    def _should_analyze_current_frame(self) -> bool:
        if self.latest_frame is None:
            return True

        image_np = self.latest_frame
        try:
            if self.pipeline_mode == "detector_two_stage":
                if self.yolo_detector is None:
                    return True
                frame_bgr = image_np[:, :, ::-1]
                detections = self.yolo_detector.detect(frame_bgr)
                return self._has_valid_detector_candidate(frame_bgr, detections)

            if self.pipeline_mode == "hailo_detector_two_stage":
                if self.hailo_detector is None:
                    return True
                h, w = image_np.shape[:2]
                model_h, model_w, _channels = self.hailo_detector.input_shape
                resized = cv2.resize(image_np, (model_w, model_h))
                hailo_output = self.hailo_detector.run(resized)
                detections = self._parse_hailo_detector_output(hailo_output, w, h)
                return self._has_valid_detector_candidate(image_np[:, :, ::-1], detections)

            if self.pipeline_mode in {"rfdetr", "two_stage"}:
                if self.rfdetr is None:
                    return True
                detections = self.rfdetr.all_detections(Image.fromarray(image_np))
                return any(bbox is not None for _name, _conf, bbox in detections)
        except Exception as exc:
            self._log_detection(f"precheck_error pipeline={self.pipeline_mode} error={exc}")
            return True

        return True

    def _has_valid_detector_candidate(self, frame_bgr: np.ndarray, detections: list[tuple]) -> bool:
        h, w = frame_bgr.shape[:2]
        ranked_detections = sorted(detections, key=lambda item: float(item[4]), reverse=True)
        for x1, y1, x2, y2, det_conf, _cls in ranked_detections[:MAX_CLASSIFIED_DETECTIONS]:
            if float(det_conf) < MIN_DETECTION_CONFIDENCE:
                continue
            if self.config.ignore_hands and self._looks_like_hand(frame_bgr, x1, y1, x2, y2):
                continue

            pad = 8
            crop = frame_bgr[
                max(0, y1 - pad): min(h, y2 + pad),
                max(0, x1 - pad): min(w, x2 + pad),
            ]
            if crop.size == 0:
                continue
            return True
        return False

    def _show_no_detection_idle_state(self) -> None:
        self._last_bboxes = []
        if self._awaiting_throw_bin is not None:
            return
        if self.led:
            try:
                self.led.send_command("idle")
            except Exception:
                pass
        self.prediction_var.set("GEEN PRODUCT")
        self.prediction_overlay.config(fg=C_SUBTEXT)
        self.prediction_overlay.place(x=10, y=10)
        self.confidence_var.set("")
        self.confidence_overlay.place_forget()
        self._highlight_bin(None)
        self.set_status("Wachten op product...", C_SUBTEXT)

    def _on_throw_timeout(self) -> None:
        """Called after 6s if no throw was detected — reset LEDs to idle."""
        self._throw_timeout_id = None
        if self._awaiting_throw_bin is None:
            return
        print(f"[DEBUG] 6s timeout — reset naar idle", flush=True)
        self._awaiting_throw_bin = None
        if self.led:
            try:
                self.led.send_command("idle")
            except Exception:
                pass
        self.prediction_var.set("GEEN PRODUCT")
        self.prediction_overlay.config(fg=C_SUBTEXT)
        self.prediction_overlay.place(x=10, y=10)
        self.confidence_var.set("")
        self.confidence_overlay.place_forget()
        self._highlight_bin(None)
        self.set_status("Wachten op product...", C_SUBTEXT)

    def _classify_worker(self, analyze_full_frame: bool = False) -> None:
        try:
            if self.latest_frame is not None:
                image_np = self.latest_frame.copy()
            elif self.camera is not None:
                image_np = self.camera.capture_array()[:, :, ::-1]
            else:
                raise RuntimeError("Geen cameraframe beschikbaar")

            pil_img = Image.fromarray(image_np)
            start   = time.time()

            if analyze_full_frame:
                result = self._run_manual_full_frame_analysis(pil_img, image_np)
            elif self.pipeline_mode == "hailo_detector_two_stage":
                result = self._run_hailo_detector_then_two_stage(image_np)
            elif self.pipeline_mode == "detector_two_stage":
                result = self._run_detector_then_two_stage(image_np)
            elif self.pipeline_mode == "rfdetr":
                result = self._run_rfdetr(pil_img)
            elif self.pipeline_mode == "hailo":
                result = self._run_hailo(image_np)
            elif self.pipeline_mode == "two_stage":
                result = self._run_detect_then_two_stage(pil_img, image_np)
            else:
                result = self._run_single_stage(image_np)

            ms = (time.time() - start) * 1000
            self.result_queue.put(("result", (*result, ms, pil_img)))

        except Exception as exc:
            self.result_queue.put(("error", str(exc)))
        finally:
            self.result_queue.put(("done", None))

    def _run_manual_full_frame_analysis(self, pil_img: "Image.Image", image_np: np.ndarray) -> tuple:
        if self.pipeline_mode in {"hailo_detector_two_stage", "detector_two_stage", "two_stage"}:
            return self._run_two_stage(image_np)
        if self.pipeline_mode == "rfdetr":
            return self._run_rfdetr(pil_img)
        if self.pipeline_mode == "hailo":
            return self._run_hailo(image_np)
        return self._run_single_stage(image_np)

    def _save_capture_photo(self, pil_img: "Image.Image") -> None:
        try:
            out_dir = Path(__file__).resolve().parent / "captures"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            ms = int((time.time() % 1) * 1000)
            out_path = out_dir / f"capture_{ts}_{ms:03d}.jpg"
            pil_img.save(out_path, format="JPEG", quality=90)
        except Exception:
            # Foto-opslag mag inferentie nooit blokkeren.
            pass

    # result format: (bin_key, label, confidence, bboxes_norm_list)
    def _run_rfdetr(self, pil_img: "Image.Image") -> tuple:
        if self.rfdetr is None:
            raise RuntimeError("RF-DETR niet geladen")
        all_det = self.rfdetr.all_detections(pil_img)
        if not all_det:
            return "rest", "Niets gevonden", 0.0, []
        # Sort by confidence descending, pick best
        all_det.sort(key=lambda x: x[1], reverse=True)
        best_name, best_conf, best_bbox = all_det[0]
        bboxes = [(name, conf, bbox) for name, conf, bbox in all_det if bbox is not None]
        bin_key = label_to_bin(best_name)
        return bin_key, best_name, best_conf, bboxes

    def _run_hailo(self, image_np: np.ndarray) -> tuple:
        if self.hailo is None:
            raise RuntimeError("Hailo niet geladen")
        arr  = self._preprocess_nhwc(image_np, self.hailo.input_shape)
        raw  = self.hailo.infer(arr)
        prob = self._to_probs(raw)
        idx  = int(np.argmax(prob))
        label= self.classes[idx] if idx < len(self.classes) else self.default_fallback
        conf = float(prob[idx])
        bin_key = label_to_bin(label)
        return bin_key, label, conf, []

    def _run_single_stage(self, image_np: np.ndarray) -> tuple:
        if self.session is None:
            raise RuntimeError("ONNX model niet geladen")
        arr   = self._preprocess_nchw(image_np, self.input_shape)
        out   = self.session.run(None, {self.input_name: arr})
        prob  = self._to_probs(np.asarray(out[0], dtype=np.float32))
        idx   = int(np.argmax(prob))
        label = self.classes[idx] if idx < len(self.classes) else self.default_fallback
        conf  = float(prob[idx])
        bin_key = label_to_bin(label)
        return bin_key, label, conf, []

    def _run_two_stage(self, image_np: np.ndarray) -> tuple:
        if self.stage1_session is None:
            raise RuntimeError("Stage1 niet geladen")
        arr1  = self._preprocess_nchw(image_np, self.stage1_input_shape)
        out1  = self.stage1_session.run(None, {self.stage1_input_name: arr1})
        prob1 = self._to_probs(np.asarray(out1[0], dtype=np.float32))
        idx1  = int(np.argmax(prob1)); conf1 = float(prob1[idx1])
        lbl1  = self.stage1_classes[idx1] if idx1 < len(self.stage1_classes) else self.default_fallback

        if conf1 < self.stage1_confidence_threshold:
            final_lbl, final_conf = self.default_fallback, conf1
        elif lbl1 == self.main_label_for_stage2 and self.stage2_session is not None:
            arr2  = self._preprocess_nchw(image_np, self.stage2_input_shape)
            out2  = self.stage2_session.run(None, {self.stage2_input_name: arr2})
            prob2 = self._to_probs(np.asarray(out2[0], dtype=np.float32))
            idx2  = int(np.argmax(prob2)); conf2 = float(prob2[idx2])
            sub   = self.stage2_overige_classes[idx2] if idx2 < len(self.stage2_overige_classes) else ""
            if conf2 < self.stage2_confidence_threshold:
                final_lbl, final_conf = self.main_label_for_stage2, conf2
            else:
                full = sub if sub.startswith(f"{self.main_label_for_stage2}/") else f"{self.main_label_for_stage2}/{sub}"
                final_lbl, final_conf = full, conf2
        else:
            final_lbl, final_conf = lbl1, conf1

        bin_key = label_to_bin(final_lbl)
        return bin_key, final_lbl, final_conf, []

    def _run_detect_then_two_stage(self, pil_img: "Image.Image", image_np: np.ndarray) -> tuple:
        """
        Gewenste flow:
        1) objectdetectie (RF-DETR) als gate
        2) classificatie met stage1 + stage2
        """
        if self.rfdetr is None:
            # Fallback als RF-DETR niet beschikbaar is.
            return self._run_two_stage(image_np)

        all_det = self.rfdetr.all_detections(pil_img)
        if not all_det:
            return "rest", "Niets gevonden", 0.0, []

        all_det.sort(key=lambda x: x[1], reverse=True)
        bboxes = [(name, conf, bbox) for name, conf, bbox in all_det if bbox is not None]
        bin_key, final_lbl, final_conf, _ = self._run_two_stage(image_np)
        return bin_key, final_lbl, final_conf, bboxes

    def _run_detector_then_two_stage(self, image_np: np.ndarray) -> tuple:
        if self.yolo_detector is None or self.two_stage_classifier is None:
            raise RuntimeError("Detector of two-stage classifier niet geladen")
        bgr = image_np[:, :, ::-1]
        detections = self.yolo_detector.detect(bgr)
        self._log_detection(f"source=onnx_detector raw_detections={len(detections)}")
        return self._classify_detector_crops(bgr, detections)

    def _run_hailo_detector_then_two_stage(self, image_np: np.ndarray) -> tuple:
        if self.hailo_detector is None or self.two_stage_classifier is None:
            raise RuntimeError("Hailo detector of two-stage classifier niet geladen")
        h, w = image_np.shape[:2]
        model_h, model_w, _channels = self.hailo_detector.input_shape
        resized = cv2.resize(image_np, (model_w, model_h))
        hailo_output = self.hailo_detector.run(resized)
        detections = self._parse_hailo_detector_output(hailo_output, w, h)
        self._log_detection(f"source=hailo_detector raw_detections={len(detections)}")
        bgr = image_np[:, :, ::-1]
        return self._classify_detector_crops(bgr, detections)

    def _classify_detector_crops(self, frame_bgr: np.ndarray, detections: list[tuple]) -> tuple:
        if not detections:
            self._log_detection("result=none reason=no_detections_after_detector")
            return "rest", "Niets gevonden", 0.0, []

        h, w = frame_bgr.shape[:2]
        best_result: tuple[str, str, float] | None = None
        best_score = -1.0
        best_bbox: tuple[str, float, tuple | None] | None = None

        ranked_detections = sorted(detections, key=lambda item: float(item[4]), reverse=True)
        self._log_detection(
            f"ranked_candidates={len(ranked_detections)} inspected={min(len(ranked_detections), MAX_CLASSIFIED_DETECTIONS)}"
        )

        for x1, y1, x2, y2, det_conf, _cls in ranked_detections[:MAX_CLASSIFIED_DETECTIONS]:
            box_txt = f"box=({x1},{y1},{x2},{y2}) det_conf={float(det_conf):.3f}"
            if float(det_conf) < MIN_DETECTION_CONFIDENCE:
                self._log_detection(f"candidate_skip reason=det_conf_too_low {box_txt}")
                continue
            if self.config.ignore_hands and self._looks_like_hand(frame_bgr, x1, y1, x2, y2):
                self._log_detection(f"candidate_skip reason=hand_filter {box_txt}")
                continue

            pad = 8
            crop = frame_bgr[
                max(0, y1 - pad): min(h, y2 + pad),
                max(0, x1 - pad): min(w, x2 + pad),
            ]
            if crop.size == 0:
                self._log_detection(f"candidate_skip reason=empty_crop {box_txt}")
                continue

            label, clf_conf = self.two_stage_classifier.classify(crop)
            if (
                label == self.default_fallback
                and float(clf_conf) < MIN_RESTAFVAL_CONFIDENCE
            ):
                self._log_detection(
                    f"candidate_skip reason=restafval_conf_too_low {box_txt} "
                    f"label={label} clf_conf={float(clf_conf):.3f}"
                )
                continue
            bbox = (
                float(x1) / max(1, w),
                float(y1) / max(1, h),
                float(x2) / max(1, w),
                float(y2) / max(1, h),
            )

            score = float(det_conf) * max(float(clf_conf), 0.01)
            self._log_detection(
                f"candidate_eval {box_txt} label={label} clf_conf={float(clf_conf):.3f} combined={score:.3f}"
            )
            if score >= MIN_COMBINED_SCORE and score > best_score:
                bin_key = label_to_bin(label)
                best_result = (bin_key, label, float(clf_conf))
                best_bbox = (label, float(clf_conf), bbox)
                best_score = score

        if best_result is None:
            self._log_detection("result=none reason=no_candidate_passed_filters")
            return "rest", "Niets gevonden", 0.0, []
        bboxes = [best_bbox] if best_bbox is not None and MAX_OVERLAY_BOXES > 0 else []
        self._log_detection(
            f"result=ok bin={best_result[0]} label={best_result[1]} conf={best_result[2]:.3f} score={best_score:.3f}"
        )
        return (*best_result, bboxes)

    @staticmethod
    def _looks_like_hand(frame_bgr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Conservative hand filter for close skin-colored objects near the camera."""
        h, w = frame_bgr.shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area_ratio = (bw * bh) / max(1.0, float(w * h))
        aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))
        touches_edge = (
            x1 <= HAND_EDGE_MARGIN_PX
            or y1 <= HAND_EDGE_MARGIN_PX
            or x2 >= (w - HAND_EDGE_MARGIN_PX)
            or y2 >= (h - HAND_EDGE_MARGIN_PX)
        )

        crop = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            return False

        # Combine two common skin-color masks to reduce false positives.
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 180, 255]))
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        ycrcb_mask = cv2.inRange(ycrcb, np.array([0, 135, 85]), np.array([255, 180, 135]))
        skin_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
        skin_ratio = float(np.count_nonzero(skin_mask)) / max(1.0, float(skin_mask.size))

        if skin_ratio >= HAND_SKIN_RATIO_STRONG:
            return True
        if touches_edge and skin_ratio >= HAND_SKIN_RATIO_EDGE:
            return True
        if area_ratio >= HAND_LARGE_AREA_RATIO and skin_ratio >= HAND_SKIN_RATIO_GENERAL:
            return True
        if touches_edge and area_ratio >= HAND_EDGE_AREA_RATIO and skin_ratio >= HAND_SKIN_RATIO_LOOSE_EDGE:
            return True
        if touches_edge and aspect_ratio >= 1.6 and skin_ratio >= 0.28:
            return True
        return False

    @staticmethod
    def _parse_hailo_detector_output(hailo_output, orig_w: int, orig_h: int) -> list[tuple]:
        detections: list[tuple] = []
        if hailo_output is None:
            return detections

        for detection in hailo_output:
            conf = float(detection.get_confidence())
            if conf < 0.45:
                continue
            bbox = detection.get_bbox()
            x1 = int(bbox.xmin() * orig_w)
            y1 = int(bbox.ymin() * orig_h)
            x2 = int(bbox.xmax() * orig_w)
            y2 = int(bbox.ymax() * orig_h)
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            area = bw * bh
            if area < 3000:
                continue
            if area / max(1.0, float(orig_w * orig_h)) > 0.85:
                continue
            detections.append((x1, y1, x2, y2, conf, int(detection.get_label())))

        if not detections:
            return []

        boxes = [[x1, y1, max(1, x2 - x1), max(1, y2 - y1)] for x1, y1, x2, y2, _conf, _cls in detections]
        scores = [float(conf) for _x1, _y1, _x2, _y2, conf, _cls in detections]
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.45, 0.35)
        if indices is None or len(indices) == 0:
            return []

        kept = np.array(indices).reshape(-1).astype(int).tolist()
        kept.sort(key=lambda idx: detections[idx][4], reverse=True)
        kept = kept[:3]
        return [detections[idx] for idx in kept]

    # ── Preprocessing ─────────────────────────────────────────────────────

    @staticmethod
    def _preprocess_nchw(image_np: np.ndarray, input_shape=None) -> np.ndarray:
        th, tw = 224, 224
        if input_shape and len(input_shape) == 4:
            try:
                if isinstance(input_shape[2], int) and input_shape[2] > 0:
                    th, tw = input_shape[2], input_shape[3]
            except Exception:
                pass
        img = Image.fromarray(image_np).resize((tw, th))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return np.expand_dims(arr.transpose(2, 0, 1), axis=0).astype(np.float32)

    @staticmethod
    def _preprocess_nhwc(image_np: np.ndarray, input_shape=None) -> np.ndarray:
        th, tw = 224, 224
        if input_shape and len(input_shape) >= 2:
            try:
                th, tw = int(input_shape[0]), int(input_shape[1])
            except Exception:
                pass
        img = Image.fromarray(image_np).resize((tw, th))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return np.expand_dims(arr, axis=0).astype(np.float32)

    @staticmethod
    def _to_probs(raw: np.ndarray) -> np.ndarray:
        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            raise RuntimeError("Lege model output")
        if np.all(arr >= 0.0) and 0.98 <= float(arr.sum()) <= 1.02:
            return arr
        exp_x = np.exp(arr - np.max(arr))
        return exp_x / max(float(exp_x.sum()), 1e-9)

    # ── Worker messages ────────────────────────────────────────────────────

    def _process_worker_messages(self) -> None:
        if not self.running:
            return
        while True:
            try:
                mtype, payload = self.result_queue.get_nowait()
            except Empty:
                break

            if mtype == "result":
                bin_key, label, conf, bboxes, ms, pil_img = payload
                self._show_results(bin_key, label, conf, bboxes, ms, pil_img)

            elif mtype == "error":
                self.set_status(f"Fout: {payload}", C_ERROR)

            elif mtype == "init_ok":
                d = payload
                if not isinstance(d, dict):
                    self.set_status(f"Init payload ongeldig type: {type(d).__name__}", C_ERROR)
                    continue
                self.pipeline_mode = str(d.get("pipeline_mode", "single"))
                self.classes       = list(d.get("classes", DEFAULT_CLASSES))
                self.hailo         = d.get("hailo")
                self.hailo_detector = d.get("hailo_detector")
                self.rfdetr        = d.get("rfdetr")
                self.yolo_detector = d.get("yolo_detector")
                self.two_stage_classifier = d.get("two_stage_classifier")
                self.session       = d.get("session")
                self.input_name    = d.get("input_name")
                self.input_shape   = d.get("input_shape")
                self.output_names  = list(d.get("output_names", []))
                self.stage1_session    = d.get("stage1_session")
                self.stage1_input_name = d.get("stage1_input_name")
                self.stage1_input_shape= d.get("stage1_input_shape")
                self.stage2_session    = d.get("stage2_session")
                self.stage2_input_name = d.get("stage2_input_name")
                self.stage2_input_shape= d.get("stage2_input_shape")
                self.stage1_classes    = list(d.get("stage1_classes", self.stage1_classes))
                self.stage2_overige_classes = list(d.get("stage2_overige_classes", self.stage2_overige_classes))
                self.main_label_for_stage2  = str(d.get("main_label_for_stage2", self.main_label_for_stage2))
                self.default_fallback       = str(d.get("default_fallback", self.default_fallback))
                self.stage1_confidence_threshold = float(d.get("stage1_confidence_threshold", 0.40))
                self.stage2_confidence_threshold = float(d.get("stage2_confidence_threshold", 0.45))
                self.camera = d.get("camera")
                self.led    = d.get("led")
                self.initialized = True
                self.auto_init_retry_count = 0

                engine_tag = {
                    "hailo_detector_two_stage": "Hailo detector + 2-stage",
                    "detector_two_stage": "YOLO ONNX + 2-stage",
                    "hailo": "Hailo AI Hat+",
                    "rfdetr": "RF-DETR (CPU)",
                    "two_stage": "ONNX 2-stage",
                    "single": "ONNX",
                }.get(self.pipeline_mode, self.pipeline_mode)
                model_name = str(d.get("resolved_model", "onbekend"))
                self.set_status(f"Klaar – {engine_tag}  |  {model_name}", C_SUCCESS)
                self.update_ui_state(enabled=True)
                self.prediction_var.set("")
                if self.led:
                    self.led.send_command("idle")

            elif mtype == "init_error":
                self.initialized = False
                short = str(payload).strip().splitlines()[0]
                self.set_status(f"Init fout: {short}", C_ERROR)
                self.auto_init_retry_count += 1
                if self.auto_init_retry_count <= self.max_auto_init_retries:
                    self.root.after(self.init_retry_delay_ms, self._retry_init_if_needed)
                else:
                    self.set_status("Init blijft falen – druk Reset om opnieuw te proberen.", C_ERROR)

            elif mtype == "init_finished":
                self.init_in_progress = False

            elif mtype == "done":
                self.worker_active = False
                self.update_ui_state(self.initialized)

        self.root.after(50, self._process_worker_messages)

    def _retry_init_if_needed(self) -> None:
        if not self.running or self.initialized or self.init_in_progress:
            return
        self.set_status("Init opnieuw proberen...", C_ACCENT)
        self.start_initialization()

    # ── Results ────────────────────────────────────────────────────────────

    def _show_results(
        self,
        bin_key: str,
        label: str,
        conf: float,
        bboxes: list,
        ms: float,
        pil_img: "Image.Image",
    ) -> None:
        # Draw bounding boxes if we have them (RF-DETR)
        self._last_bboxes = list(bboxes or [])

        if label == "Niets gevonden":
            if self._awaiting_throw_bin is None:
                self._show_no_detection_idle_state()
                self.set_status(f"{ms:.0f} ms  |  Geen product gedetecteerd", C_SUBTEXT)
            return

        # Lock state once a bin is armed — ignore new detections until throw or timeout.
        if self._awaiting_throw_bin is not None:
            return

        # Arm BEFORE LED command so no second scan can start during the subprocess call.
        self._awaiting_throw_bin = bin_key if bin_key else None
        if self._throw_timeout_id is not None:
            self.root.after_cancel(self._throw_timeout_id)
        self._throw_timeout_id = self.root.after(6_000, self._on_throw_timeout)
        print(f"[DEBUG] Gewapend op {bin_key!r}, 6s timer gestart", flush=True)

        # LED
        led_cmd = bin_to_led_cmd(bin_key)
        if self.led and led_cmd:
            led_resp = self.led.send_command(led_cmd)
            if isinstance(led_resp, str) and (led_resp.startswith("ERROR") or led_resp.startswith("UNKNOWN")):
                self.set_status(f"LED fout: {led_resp}", C_ERROR)

        # Overlay text
        if label and label != "Niets gevonden":
            display_label = label
        else:
            display_label = bin_label(bin_key) if bin_key else label
        color = {b["key"]: b["active"] for b in BIN_DEFS}.get(bin_key, C_ACCENT)

        self.prediction_var.set(display_label.upper())
        self.prediction_overlay.config(fg=color)
        self.prediction_overlay.place(x=10, y=10)

        self.confidence_var.set(f"{conf*100:.1f}%  –  {label}")
        self.confidence_overlay.place(x=10, y=56)

        self._highlight_bin(bin_key)

        engine_tag = {
            "hailo_detector_two_stage": "Hailo+2s",
            "detector_two_stage": "YOLO+2s",
            "hailo": "Hailo",
            "rfdetr": "RF-DETR",
            "two_stage": "ONNX-2s",
            "single": "ONNX",
        }.get(self.pipeline_mode, "")
        self.set_status(f"{ms:.0f} ms  |  {engine_tag}  |  LED: {led_cmd}", "#555555")

    def _update_ultrasonic_ui(self) -> None:
        if not self.running:
            return
        if not self.ultra or not self.ultra.enabled:
            for k, lbl in self._bin_fill_labels.items():
                if k == "org":
                    lbl.config(text="niveau: n.v.t.", fg=C_SUBTEXT)
                else:
                    lbl.config(text="niveau: uit", fg=C_SUBTEXT)
            self.root.after(1000, self._update_ultrasonic_ui)
            return

        snap = self.ultra.snapshot()
        for key, lbl in self._bin_fill_labels.items():
            data = snap.get(key, {})
            txt = str(data.get("text", "onbekend"))
            full = bool(data.get("is_full", False))
            if txt == "n.v.t.":
                lbl.config(text="niveau: n.v.t.", fg=C_SUBTEXT)
            elif full:
                lbl.config(text=f"niveau: {txt}", fg=C_ERROR)
            else:
                lbl.config(text=f"niveau: {txt}", fg=C_SUBTEXT)
        self.root.after(1000, self._update_ultrasonic_ui)

    @staticmethod
    def _draw_bboxes(img: "Image.Image", bboxes: list) -> "Image.Image":
        """Draw bounding boxes on a PIL image. bboxes = list of (name, conf, (x1,y1,x2,y2) normalised)."""
        out  = img.copy()
        draw = ImageDraw.Draw(out)
        w, h = out.size
        try:
            fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            fnt = ImageFont.load_default()

        color_cycle = ["#4CAF50", "#2196F3", "#FFC107", "#E91E63", "#FF5722"]
        for i, (name, conf, bbox) in enumerate(bboxes):
            if bbox is None:
                continue
            x1n, y1n, x2n, y2n = bbox
            x1, y1, x2, y2 = int(x1n*w), int(y1n*h), int(x2n*w), int(y2n*h)
            col = color_cycle[i % len(color_cycle)]
            draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
            txt = f"{name}  {conf*100:.0f}%"
            draw.rectangle([x1, y1-22, x1+len(txt)*9, y1], fill=col)
            draw.text((x1+4, y1-20), txt, fill="white", font=fnt)
        return out

    def _fit_to_preview(self, img: "Image.Image") -> "Image.Image":
        pw = self.preview_label.winfo_width()
        ph = self.preview_label.winfo_height()
        if pw <= 1 or ph <= 1:
            return img.resize((640, 480), Image.Resampling.LANCZOS)
        return ImageOps.pad(
            img,
            (pw, ph),
            method=Image.Resampling.LANCZOS,
            color=(17, 17, 17),
            centering=(0.5, 0.5),
        )

    def _highlight_bin(self, active_key: str | None) -> None:
        for b in BIN_DEFS:
            k  = b["key"]
            fr = self._bin_frames[k]
            ic = self._bin_labels[k]
            if k == active_key:
                fr.config(bg=b["active"], highlightbackground=b["border"])
                ic.config(bg=b["active"], fg="white")
            else:
                fr.config(bg=b["idle"],   highlightbackground=b["border"])
                ic.config(bg=b["idle"],   fg=b["border"])
        self._active_bin = active_key

    def reset_classification(self) -> None:
        self.worker_active = False
        self.prediction_var.set("")
        self.confidence_var.set("")
        self.prediction_overlay.place_forget()
        self.confidence_overlay.place_forget()
        self._last_bboxes = []
        self._highlight_bin(None)
        self._awaiting_throw_bin = None
        if self._throw_timeout_id is not None:
            self.root.after_cancel(self._throw_timeout_id)
            self._throw_timeout_id = None
        if self.led:
            self.led.send_command("idle")
        self.set_status("Gereset", C_SUCCESS)
        if not self.initialized and not self.init_in_progress:
            self.auto_init_retry_count = 0
            self.start_initialization()
            return
        if self.initialized:
            self.update_ui_state(enabled=True)

    def update_ui_state(self, enabled: bool) -> None:
        self.btn_classify.config(
            state="normal" if enabled else "disabled",
            bg=C_ACCENT if enabled else "#AAAAAA"
        )
        if self.initialized:
            self.btn_reset.config(state="normal")

    def set_status(self, text: str, color: str = C_SUBTEXT) -> None:
        self.status_var.set(text)
        self.status_lbl.config(fg=color)

    def toggle_fullscreen(self) -> None:
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))

    def on_closing(self) -> None:
        self.running = False
        try:
            if self.camera:
                self.camera.stop()
            if self.hailo:
                self.hailo.close()
            if self.hailo_detector:
                self.hailo_detector.close()
            if self.led:
                self.led.close()
            if self.ultra:
                self.ultra.close()
        finally:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _load_metadata(path: str | None) -> dict:
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                print(f"Metadata fout ({path}): verwacht dict, kreeg {type(data).__name__}")
                return {}
        except Exception as exc:
            print(f"Metadata fout ({path}): {exc}")
            return {}

    @staticmethod
    def _parse_threshold(meta: dict, keys: tuple, default: float) -> float:
        for k in keys:
            v = meta.get(k)
            if isinstance(v, (int, float)):
                return max(0.0, min(1.0, float(v)))
        return default

    @staticmethod
    def _build_two_stage_classes(s1_cls: list[str], s2_cls: list[str], main_label: str) -> list[str]:
        out: list[str] = []
        for lbl in s1_cls:
            if lbl == main_label:
                for sub in s2_cls:
                    out.append(sub if sub.startswith(f"{main_label}/") else f"{main_label}/{sub}")
            else:
                out.append(lbl)
        return out


# ── Entry point ────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      dest="model_path", help="Model pad (.hef, .pth, of .onnx)")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--rotate",     type=int, default=0, help="Camera rotatie in graden")
    p.add_argument("--threshold",  type=float, default=0.45, help="Detectie confidence drempel")
    p.add_argument("--ignore-hands", dest="ignore_hands", action="store_true",
                   help="Negeer hand-detecties volledig, ook in de overlay")
    p.add_argument("--show-hands", dest="ignore_hands", action="store_false",
                   help="Toon hand-achtige detecties toch in de pipeline")
    p.set_defaults(ignore_hands=True)
    p.add_argument("--no-detection-log", dest="detection_logging", action="store_false",
                   help="Schakel detectie-debuglogging naar detect_debug.log uit")
    p.set_defaults(detection_logging=True)
    p.add_argument("--ultra-debug", action="store_true",
                   help="Print debug regels voor ultrasoonsensoren")
    p.add_argument("--ultra-debug-every-cycles", type=int, default=1,
                   help="Print elke N meetcycli (1 = elke cyclus)")
    p.add_argument("--org-trig", type=int, default=None,
                   help="BCM pin voor Organisch ultrasonic TRIG")
    p.add_argument("--org-echo", type=int, default=None,
                   help="BCM pin voor Organisch ultrasonic ECHO")
    p.add_argument("--pmd-trig", type=int, default=None,
                   help="BCM pin voor PMD ultrasonic TRIG")
    p.add_argument("--pmd-echo", type=int, default=None,
                   help="BCM pin voor PMD ultrasonic ECHO")
    p.add_argument("--ultra-only", action="store_true",
                   help="Start alleen ultrasoon monitor (geen GUI vereist)")
    p.add_argument("--ultra-runtime", type=int, default=0,
                   help="Stop ultra-only mode na N seconden (0 = oneindig)")
    return p.parse_args()


def run_ultrasonic_debug_only(
    debug: bool,
    debug_every_cycles: int,
    runtime_s: int,
    org_trig: int | None = None,
    org_echo: int | None = None,
    pmd_trig: int | None = None,
    pmd_echo: int | None = None,
) -> None:
    print("[Ultrasoon] Ultra-only modus gestart")
    sensor_overrides: dict[str, dict[str, int] | None] = {}
    if org_trig is not None and org_echo is not None:
        sensor_overrides["org"] = {"trig": org_trig, "echo": org_echo}
    if pmd_trig is not None and pmd_echo is not None:
        sensor_overrides["pmd"] = {"trig": pmd_trig, "echo": pmd_echo}
    monitor = UltrasonicMonitor(
        debug=debug,
        debug_every_cycles=debug_every_cycles,
        sensor_overrides=sensor_overrides,
    )
    if not monitor.enabled:
        print("[Ultrasoon] GPIO niet beschikbaar")
        return
    start = time.time()
    try:
        while True:
            if runtime_s > 0 and (time.time() - start) >= runtime_s:
                print("[Ultrasoon] Runtime bereikt, stoppen")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Ultrasoon] Onderbroken met Ctrl+C")
    finally:
        monitor.close()
        print("[Ultrasoon] Ultra-only modus gestopt")


if __name__ == "__main__":
    args   = get_args()
    if (args.org_trig is None) ^ (args.org_echo is None):
        raise SystemExit("Gebruik --org-trig en --org-echo samen, of geen van beide.")
    if (args.pmd_trig is None) ^ (args.pmd_echo is None):
        raise SystemExit("Gebruik --pmd-trig en --pmd-echo samen, of geen van beide.")
    if args.ultra_only:
        run_ultrasonic_debug_only(
            debug=args.ultra_debug,
            debug_every_cycles=args.ultra_debug_every_cycles,
            runtime_s=args.ultra_runtime,
            org_trig=args.org_trig,
            org_echo=args.org_echo,
            pmd_trig=args.pmd_trig,
            pmd_echo=args.pmd_echo,
        )
        raise SystemExit(0)
    config = DisplayConfig(
        model_path=args.model_path,
        fullscreen=args.fullscreen,
        rotate=args.rotate,
        det_threshold=args.threshold,
        ignore_hands=args.ignore_hands,
        detection_logging=args.detection_logging,
        ultra_debug=args.ultra_debug,
        ultra_debug_every_cycles=args.ultra_debug_every_cycles,
        org_ultra_trig=args.org_trig,
        org_ultra_echo=args.org_echo,
        pmd_ultra_trig=args.pmd_trig,
        pmd_ultra_echo=args.pmd_echo,
    )
    app = InferenceGUI(config)
    app.run()

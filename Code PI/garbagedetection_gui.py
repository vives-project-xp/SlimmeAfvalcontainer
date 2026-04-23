import argparse
import json
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import onnxruntime as ort
import tkinter as tk
from tkinter import ttk
from led_controller import LedController

try:
    from picamera2 import Picamera2
except ModuleNotFoundError as exc:
    if exc.name == "libcamera":
        raise ModuleNotFoundError(
            "Module 'libcamera' ontbreekt in deze Python-omgeving.\n"
            "Op Raspberry Pi installeer je dit via apt:\n"
            "  sudo apt install python3-libcamera python3-picamera2\n"
            "Gebruik daarna de system Python of een venv met --system-site-packages."
        ) from exc
    raise

try:
    from PIL import Image, ImageTk, ImageEnhance
except ImportError as exc:
    raise ImportError(
        "PIL.ImageTk ontbreekt. Installeer in je actieve omgeving:\n"
        "  sudo apt install python3-pil.imagetk python3-tk\n"
        "of:\n"
        "  python -m pip install --upgrade pillow"
    ) from exc


DEFAULT_CLASSES = (
    "Organisch",            # Index 0
    "Overige/Batterijen",   # Index 1
    "Overige/Elektronica",  # Index 2
    "Overige/Glas",         # Index 3
    "Overige/Lightbulbs",   # Index 4
    "Overige/Metaal",       # Index 5
    "PMD",                  # Index 6
    "Papier",               # Index 7
    "Restafval",            # Index 8
)
DEFAULT_COLORS = (
    "#4CAF50",  # Organisch
    "#FF7043",  # Overige/Batterijen
    "#EF5350",  # Overige/Elektronica
    "#26A69A",  # Overige/Glas
    "#FFD54F",  # Overige/Lightbulbs
    "#8D6E63",  # Overige/Metaal
    "#FFC107",  # PMD
    "#2196F3",  # Papier
    "#757575",  # Restafval
)

# Kleurenpalet (Dark Theme)
COLOR_BG = "#1E1E1E"
COLOR_SIDEBAR = "#2D2D2D"
COLOR_TEXT = "#E0E0E0"
COLOR_ACCENT = "#3498DB"
COLOR_SUCCESS = "#2ECC71"
COLOR_ERROR = "#E74C3C"

CLASS_COLOR_MAP = {
    "Organisch": "#4CAF50",
    "PMD": "#FFC107",
    "Papier": "#2196F3",
    "Restafval": "#757575",
    "Overige": "#9E9E9E",
    "Overige/Batterijen": "#FF7043",
    "Overige/Elektronica": "#EF5350",
    "Overige/Glas": "#26A69A",
    "Overige/Lightbulbs": "#FFD54F",
    "Overige/Metaal": "#8D6E63",
}


@dataclass(frozen=True)
class DisplayConfig:
    model_path: str | None = None
    window_width: int = 1024  # Breder venster
    window_height: int = 600   # Standaard Pi 7" display
    preview_width: int = 640   # Grotere preview
    preview_height: int = 480
    fullscreen: bool = False
    rotate: int = 0
    update_ms: int = 50       # Snellere preview update


@dataclass(frozen=True)
class TwoStageConfig:
    stage1_path: str
    stage2_path: str
    metadata_path: str | None = None


def resolve_model_path(model_path: str | None = None) -> list[str]:
    """Zoek alle bruikbare ONNX-modellen, op volgorde van prioriteit."""
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    if model_path:
        user_path = Path(model_path).expanduser()
        if user_path.is_absolute():
            candidates.append(user_path)
        else:
            candidates.append((Path.cwd() / user_path).resolve())
            candidates.append((script_dir / user_path).resolve())

    # Prioriteit 1: klassiek classificatiemodel
    candidates.append((script_dir / "model.onnx").resolve())
    candidates.append((script_dir / "inference_model.onnx").resolve())
    
    # Zoek in AI submap
    ai_subdir = script_dir / "AI"
    if ai_subdir.exists():
        candidates.append((ai_subdir / "model.onnx").resolve())
        candidates.append((ai_subdir / "inference_model.onnx").resolve())

    # Zoek in Ai-model map
    ai_dir = script_dir.parent / "Ai-model"
    if ai_dir.exists():
        candidates.append((ai_dir / "model.onnx").resolve())
        candidates.append((ai_dir / "inference_model.onnx").resolve())

    # Extra fallback
    candidates.append((script_dir / "inference_model.sim.onnx").resolve())

    checked: list[Path] = []
    seen: set[str] = set()
    valid_candidates: list[str] = []

    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if candidate.is_file():
            valid_candidates.append(str(candidate))

    if not valid_candidates:
        available_models = sorted(path.name for path in script_dir.glob("*.onnx"))
        available_text = ", ".join(available_models) if available_models else "geen"
        checked_text = ", ".join(str(path) for path in checked)
        raise FileNotFoundError(
            "Geen inferences-model gevonden.\n"
            f"Geprobeerd: {checked_text}\n"
            f"Beschikbaar in {script_dir}: {available_text}"
        )
    
    return valid_candidates


def resolve_two_stage_paths(model_path: str | None = None) -> TwoStageConfig | None:
    """Zoek een two-stage modelset op bekende locaties."""
    script_dir = Path(__file__).resolve().parent
    search_dirs: list[Path] = []

    if model_path:
        user_path = Path(model_path).expanduser()
        if user_path.is_file():
            if user_path.name != "stage1_main.onnx":
                return None
            stage1 = user_path.resolve()
            stage2 = (stage1.parent / "stage2_overige.onnx").resolve()
            metadata = (stage1.parent / "two_stage_metadata.json").resolve()
            if stage2.is_file():
                return TwoStageConfig(
                    stage1_path=str(stage1),
                    stage2_path=str(stage2),
                    metadata_path=str(metadata) if metadata.is_file() else None,
                )
            return None

        if user_path.is_dir():
            if user_path.is_absolute():
                search_dirs.append(user_path.resolve())
            else:
                search_dirs.append((Path.cwd() / user_path).resolve())
                search_dirs.append((script_dir / user_path).resolve())

    search_dirs.extend(
        [
            (script_dir / "AI").resolve(),
            script_dir.resolve(),
            (script_dir.parent / "Ai-model").resolve(),
        ]
    )

    seen: set[str] = set()
    for root in search_dirs:
        root_key = str(root)
        if root_key in seen:
            continue
        seen.add(root_key)
        stage1 = root / "stage1_main.onnx"
        stage2 = root / "stage2_overige.onnx"
        if stage1.is_file() and stage2.is_file():
            metadata = root / "two_stage_metadata.json"
            return TwoStageConfig(
                stage1_path=str(stage1),
                stage2_path=str(stage2),
                metadata_path=str(metadata) if metadata.is_file() else None,
            )

    return None



class InferenceGUI:
    def __init__(self, config: DisplayConfig):
        self.config = config
        self.classes = list(DEFAULT_CLASSES)
        self.colors = list(DEFAULT_COLORS)
        self.running = True
        self.initialized = False
        self.worker_active = False
        self.init_in_progress = False
        self.init_retry_delay_ms = 10000
        self.max_auto_init_retries = 6
        self.auto_init_retry_count = 0
        self.result_queue: Queue[tuple[str, object]] = Queue()
        self.latest_frame: np.ndarray | None = None
        self.pipeline_mode = "single"
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.input_shape = None
        self.output_names: list[str] = []
        self.stage1_session: ort.InferenceSession | None = None
        self.stage1_input_name: str | None = None
        self.stage1_input_shape = None
        self.stage2_session: ort.InferenceSession | None = None
        self.stage2_input_name: str | None = None
        self.stage2_input_shape = None
        self.stage1_classes: list[str] = ["Organisch", "PMD", "Papier", "Restafval", "Overige"]
        self.stage2_overige_classes: list[str] = ["Batterijen", "Elektronica", "Glas", "Lightbulbs", "Metaal"]
        self.main_label_for_stage2 = "Overige"
        self.default_fallback = "Restafval"
        self.stage1_confidence_threshold = 0.40
        self.stage2_confidence_threshold = 0.45
        self.camera: Picamera2 | None = None
        self.led: LedController | None = None

        self.setup_ui()
        self.update_ui_state(enabled=False)
        self.set_status("Systeem opstarten...", COLOR_ACCENT)
        self.update_preview()
        
        # Start initialisatie in achtergrond
        self.start_initialization()

    def start_initialization(self) -> None:
        if not self.running or self.init_in_progress or self.initialized:
            return

        self.init_in_progress = True
        init_thread = threading.Thread(target=self._initialize_worker)
        init_thread.daemon = True
        init_thread.start()

    def setup_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("AI Waste Classifier")
        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")
        self.root.configure(bg=COLOR_BG)

        if self.config.fullscreen:
            self.root.attributes("-fullscreen", True)
        
        self.root.bind("<Escape>", lambda _event: self.toggle_fullscreen())
        self.root.bind("<F11>", lambda _event: self.toggle_fullscreen())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Styling
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Horizontal.TProgressbar", thickness=20, troughcolor=COLOR_SIDEBAR, background=COLOR_ACCENT)

        # Main Layout: 2 kolommen
        # Links: Camera preview (groot)
        # Rechts: Info & Controls
        
        main_container = tk.Frame(self.root, bg=COLOR_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left Column (Camera)
        left_col = tk.Frame(main_container, bg="black", width=self.config.preview_width, height=self.config.preview_height)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_col.pack_propagate(False) # Forceer grootte

        self.preview_label = tk.Label(
            left_col,
            bg="black",
            fg="white",
            text="Camera laden...",
            font=("Helvetica", 16)
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # Right Column (Controls & Results)
        right_col = tk.Frame(main_container, bg=COLOR_SIDEBAR, width=300)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_col.pack_propagate(False)

        # Titel
        title_lbl = tk.Label(
            right_col,
            text="Slimme\nAfvalcontainer",
            font=("Helvetica", 20, "bold"),
            bg=COLOR_SIDEBAR,
            fg="white",
            justify="center"
        )
        title_lbl.pack(pady=(20, 30))

        # Resultaat Display
        self.prediction_var = tk.StringVar(value="Gereed")
        self.prediction_label = tk.Label(
            right_col,
            textvariable=self.prediction_var,
            font=("Helvetica", 24, "bold"),
            bg=COLOR_SIDEBAR,
            fg=COLOR_ACCENT,
            wraplength=280
        )
        self.prediction_label.pack(pady=(0, 10))

        self.confidence_var = tk.StringVar(value="-- %")
        conf_lbl = tk.Label(
            right_col,
            textvariable=self.confidence_var,
            font=("Helvetica", 14),
            bg=COLOR_SIDEBAR,
            fg="#AAAAAA"
        )
        conf_lbl.pack(pady=(0, 30))

        # Knoppen
        btn_frame = tk.Frame(right_col, bg=COLOR_SIDEBAR)
        btn_frame.pack(fill=tk.X, padx=20, pady=10)

        self.btn_classify = tk.Button(
            btn_frame,
            text="ANALYSEER NU",
            command=self.classify_threaded,
            font=("Helvetica", 14, "bold"),
            bg=COLOR_ACCENT,
            fg="white",
            activebackground="#2980B9",
            activeforeground="white",
            bd=0,
            padx=10,
            pady=15,
            cursor="hand2"
        )
        self.btn_classify.pack(fill=tk.X, pady=10)

        # Extra ruimte tussen de knoppen
        spacer = tk.Frame(btn_frame, bg=COLOR_SIDEBAR, height=30)
        spacer.pack()

        self.btn_reset = tk.Button(
            btn_frame,
            text="Reset",
            command=self.reset_classification,
            font=("Helvetica", 10),
            bg="#555555",
            fg="white",
            activebackground="#444444",
            activeforeground="white",
            bd=0,
            padx=8,
            pady=8,
            cursor="hand2"
        )
        self.btn_reset.pack(pady=5)

        # Status Balk (onderaan)
        self.status_var = tk.StringVar(value="")
        status_bar = tk.Label(
            right_col,
            textvariable=self.status_var,
            font=("Helvetica", 10),
            bg=COLOR_SIDEBAR,
            fg="#888888",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padx=10
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.after(50, self._process_worker_messages)

    def _initialize_worker(self) -> None:
        try:
            print("Initializing model...")
            runtime_payload: dict[str, object] = {}
            two_stage = resolve_two_stage_paths(self.config.model_path)
            two_stage_error: Exception | None = None

            if two_stage is not None:
                try:
                    print(f"Two-stage gevonden: {two_stage.stage1_path} + {two_stage.stage2_path}")
                    stage1_session = ort.InferenceSession(two_stage.stage1_path)
                    stage2_session = ort.InferenceSession(two_stage.stage2_path)

                    stage1_input_meta = stage1_session.get_inputs()[0]
                    stage2_input_meta = stage2_session.get_inputs()[0]

                    metadata = self._load_two_stage_metadata(two_stage.metadata_path)
                    stage1_classes = list(metadata.get("stage1_classes") or self.stage1_classes)
                    stage2_classes = list(metadata.get("stage2_overige_classes") or self.stage2_overige_classes)
                    main_label = str(metadata.get("main_label_for_stage2") or self.main_label_for_stage2)
                    default_fallback = str(metadata.get("default_fallback") or self.default_fallback)
                    stage1_th = self._parse_threshold(
                        metadata,
                        keys=("stage1_confidence_threshold", "stage1_threshold", "stage1_conf_threshold"),
                        default=self.stage1_confidence_threshold,
                    )
                    stage2_th = self._parse_threshold(
                        metadata,
                        keys=("stage2_confidence_threshold", "stage2_threshold", "stage2_conf_threshold"),
                        default=self.stage2_confidence_threshold,
                    )

                    combined_classes = self._build_two_stage_classes(stage1_classes, stage2_classes, main_label)
                    runtime_payload = {
                        "pipeline_mode": "two_stage",
                        "stage1_session": stage1_session,
                        "stage1_input_name": stage1_input_meta.name,
                        "stage1_input_shape": stage1_input_meta.shape,
                        "stage2_session": stage2_session,
                        "stage2_input_name": stage2_input_meta.name,
                        "stage2_input_shape": stage2_input_meta.shape,
                        "stage1_classes": stage1_classes,
                        "stage2_overige_classes": stage2_classes,
                        "main_label_for_stage2": main_label,
                        "default_fallback": default_fallback,
                        "stage1_confidence_threshold": stage1_th,
                        "stage2_confidence_threshold": stage2_th,
                        "classes": combined_classes,
                        "resolved_model": f"{Path(two_stage.stage1_path).name} + {Path(two_stage.stage2_path).name}",
                        "output_names": [o.name for o in stage1_session.get_outputs()],
                    }
                except Exception as exc:
                    two_stage_error = exc
                    print(f"Two-stage laden mislukt: {exc}")
                    print("Fallback naar single-stage model...")

            if not runtime_payload:
                candidates = resolve_model_path(self.config.model_path)
                session = None
                resolved_model = None
                last_error = None
                input_name = None
                input_shape = None

                for model_path in candidates:
                    try:
                        print(f"Trying to load model: {model_path}")
                        sess = ort.InferenceSession(model_path)
                        input_meta = sess.get_inputs()[0]
                        input_name = input_meta.name
                        input_shape = input_meta.shape
                        session = sess
                        resolved_model = model_path
                        print(f"Successfully loaded: {model_path}")
                        print(f"Model Input Name: {input_name}, Shape: {input_shape}")
                        break
                    except Exception as exc:
                        print(f"Failed to load {model_path}: {exc}")
                        last_error = exc

                if session is None:
                    if two_stage_error is not None:
                        raise RuntimeError(
                            "Two-stage model laden mislukt en single-stage fallback niet gevonden.\n"
                            f"Two-stage fout: {two_stage_error}\n"
                            f"Single-stage fout: {last_error}"
                        )
                    raise last_error or RuntimeError("Geen geldig ONNX-model gevonden/geladen.")

                runtime_payload = {
                    "pipeline_mode": "single",
                    "session": session,
                    "input_name": input_name,
                    "input_shape": input_shape,
                    "output_names": [o.name for o in session.get_outputs()],
                    "classes": list(DEFAULT_CLASSES),
                    "resolved_model": str(resolved_model),
                }

            print("Initializing camera...")
            camera = self._initialize_camera_with_retry()

            print("Initializing LED controller...")
            led = LedController()

            if not self.running:
                camera.stop()
                led.close()
                return

            runtime_payload["camera"] = camera
            runtime_payload["led"] = led
            self.result_queue.put(("init_ok", runtime_payload))
        except Exception as exc:
            self.result_queue.put(("init_error", f"{exc}\n{traceback.format_exc()}"))
        finally:
            self.result_queue.put(("init_finished", None))

    @staticmethod
    def _load_two_stage_metadata(metadata_path: str | None) -> dict:
        if not metadata_path:
            return {}
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            print(f"Metadata kon niet geladen worden ({metadata_path}): {exc}")
            return {}

    @staticmethod
    def _parse_threshold(metadata: dict, keys: tuple[str, ...], default: float) -> float:
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, (int, float)):
                return max(0.0, min(1.0, float(value)))
        return default

    @staticmethod
    def _build_two_stage_classes(stage1_classes: list[str], stage2_classes: list[str], main_label_for_stage2: str) -> list[str]:
        classes: list[str] = []
        for label in stage1_classes:
            if label == main_label_for_stage2:
                for sub in stage2_classes:
                    combined = sub if sub.startswith(f"{main_label_for_stage2}/") else f"{main_label_for_stage2}/{sub}"
                    classes.append(combined)
            else:
                classes.append(label)
        return classes

    def _initialize_camera_with_retry(self, attempts: int = 5, base_wait_s: float = 1.5) -> Picamera2:
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            camera = None
            try:
                camera = Picamera2()
                # Gebruik maximale resolutie voor betere kwaliteit, resize voor preview
                camera_config = camera.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                camera.configure(camera_config)
                camera.start()
                time.sleep(1)
                print(f"Camera init geslaagd op poging {attempt}/{attempts}")
                return camera
            except Exception as exc:
                last_error = exc
                wait_s = base_wait_s * attempt
                print(f"Camera init poging {attempt}/{attempts} mislukt: {exc}")
                if camera is not None:
                    try:
                        camera.stop()
                    except Exception:
                        pass
                if attempt < attempts:
                    time.sleep(wait_s)

        raise RuntimeError(f"Camera initialisatie mislukt na {attempts} pogingen: {last_error}")

    def update_preview(self) -> None:
        if not self.running:
            return

        if self.camera is None:
            self.root.after(self.config.update_ms, self.update_preview)
            return

        try:
            image = self.camera.capture_array()
        except Exception as exc:
            self.set_status(f"Camera fout: {exc}", COLOR_ERROR)
            self.root.after(self.config.update_ms, self.update_preview)
            return

        # BGR naar RGB
        image = image[:, :, ::-1]
        self.latest_frame = image.copy()
        
        img = Image.fromarray(image)
        if self.config.rotate:
            img = img.rotate(self.config.rotate, expand=True)

        # Slim schalen naar preview venster met behoud van aspect ratio of 'cover'
        preview_w = self.preview_label.winfo_width()
        preview_h = self.preview_label.winfo_height()
        
        if preview_w > 1 and preview_h > 1:
             img = img.resize((preview_w, preview_h), Image.Resampling.LANCZOS)
        else:
             img = img.resize((self.config.preview_width, self.config.preview_height), Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=photo, text="")
        self.preview_label.image = photo # Keep reference
        
        self.root.after(self.config.update_ms, self.update_preview)

    def preprocess_image(self, image: np.ndarray, input_shape=None) -> np.ndarray:
        # Default target
        target_h, target_w = 224, 224
        
        # Try to detect dynamic shape from model metadata
        if input_shape:
            try:
                # Typically [batch, channels, height, width] or [batch, height, width, channels]
                shape = input_shape
                if len(shape) == 4:
                     # Check for NCHW format (most common for ONNX/PyTorch)
                     # shape[2] and shape[3] are likely H/W
                     if isinstance(shape[2], int) and isinstance(shape[3], int):
                         if shape[2] > 0 and shape[3] > 0:
                            target_h, target_w = shape[2], shape[3]
                     # Check for NHWC (TensorFlow style)
                     # shape[1] and shape[2] are likely H/W if channel is last
                     elif isinstance(shape[1], int) and isinstance(shape[2], int):
                         if shape[1] > 0 and shape[2] > 0 and (shape[3] == 3 or shape[3] == 1):
                            target_h, target_w = shape[1], shape[2]
            except Exception:
                pass # Fallback to default 224x224
        
        img = Image.fromarray(image).resize((target_w, target_h))
        img_array = np.array(img).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        return img_array

    def classify_threaded(self) -> None:
        if not self.running or self.worker_active:
            return
        if not self.initialized:
            return

        self.worker_active = True
        self.update_ui_state(enabled=False)
        self.set_status("Analyseren...", COLOR_ACCENT)
        self.prediction_var.set("Bezig...")
        self.confidence_var.set("")
        
        thread = threading.Thread(target=self._classify_worker)
        thread.daemon = True
        thread.start()

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

    def _index_for_label(self, label: str) -> int:
        if label in self.classes:
            return self.classes.index(label)
        self.classes.append(label)
        self.colors.append(CLASS_COLOR_MAP.get(label, COLOR_TEXT))
        return len(self.classes) - 1

    def _run_single_stage(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        if self.session is None or self.input_name is None:
            raise RuntimeError("Model niet geladen")

        img_array = self.preprocess_image(image, self.input_shape)
        outputs = self.session.run(None, {self.input_name: img_array})
        probs = self._to_probabilities(np.asarray(outputs[0], dtype=np.float32))
        idx = int(np.argmax(probs))
        return probs, idx

    def _run_two_stage(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        if self.stage1_session is None or self.stage1_input_name is None:
            raise RuntimeError("Stage 1 model niet geladen")

        img1 = self.preprocess_image(image, self.stage1_input_shape)
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
            img2 = self.preprocess_image(image, self.stage2_input_shape)
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
                final_label = sub_label if sub_label.startswith(f"{self.main_label_for_stage2}/") else f"{self.main_label_for_stage2}/{sub_label}"
                final_conf = conf2

        probabilities = np.zeros(len(self.classes), dtype=np.float32)
        final_idx = self._index_for_label(final_label)
        if final_idx >= len(probabilities):
            probabilities = np.zeros(len(self.classes), dtype=np.float32)
        probabilities[final_idx] = final_conf
        return probabilities, final_idx

    def _classify_worker(self) -> None:
        try:
            if self.latest_frame is not None:
                image = self.latest_frame.copy()
            elif self.camera is not None:
                image = self.camera.capture_array()
            else:
                raise RuntimeError("Geen beeld")

            start = time.time()
            if self.pipeline_mode == "two_stage":
                probabilities, predicted_idx = self._run_two_stage(image)
            else:
                probabilities, predicted_idx = self._run_single_stage(image)
            inference_time = (time.time() - start) * 1000

            self.result_queue.put(("result", (probabilities, predicted_idx, inference_time)))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))
        finally:
            self.result_queue.put(("done", None))

    def _process_worker_messages(self) -> None:
        if not self.running:
            return

        while True:
            try:
                message_type, payload = self.result_queue.get_nowait()
            except Empty:
                break

            if message_type == "result":
                probabilities, predicted_idx, inference_time = payload
                self._show_results(probabilities, predicted_idx, inference_time)
            elif message_type == "error":
                self.set_status(f"Fout: {payload}", COLOR_ERROR)
            elif message_type == "init_ok":
                data = payload
                self.pipeline_mode = str(data.get("pipeline_mode", "single"))
                self.classes = list(data.get("classes", DEFAULT_CLASSES))
                self.colors = [CLASS_COLOR_MAP.get(name, COLOR_TEXT) for name in self.classes]

                self.session = data.get("session")
                self.input_name = data.get("input_name")
                self.input_shape = data.get("input_shape")
                self.output_names = list(data.get("output_names", []))

                self.stage1_session = data.get("stage1_session")
                self.stage1_input_name = data.get("stage1_input_name")
                self.stage1_input_shape = data.get("stage1_input_shape")
                self.stage2_session = data.get("stage2_session")
                self.stage2_input_name = data.get("stage2_input_name")
                self.stage2_input_shape = data.get("stage2_input_shape")
                self.stage1_classes = list(data.get("stage1_classes", self.stage1_classes))
                self.stage2_overige_classes = list(data.get("stage2_overige_classes", self.stage2_overige_classes))
                self.main_label_for_stage2 = str(data.get("main_label_for_stage2", self.main_label_for_stage2))
                self.default_fallback = str(data.get("default_fallback", self.default_fallback))
                self.stage1_confidence_threshold = float(data.get("stage1_confidence_threshold", self.stage1_confidence_threshold))
                self.stage2_confidence_threshold = float(data.get("stage2_confidence_threshold", self.stage2_confidence_threshold))

                self.camera = data.get("camera")
                self.led = data.get("led")
                self.initialized = True
                self.auto_init_retry_count = 0

                resolved_model = str(data.get("resolved_model", "onbekend"))
                status_text = f"Klaar ({self.pipeline_mode}). Model: {resolved_model}"
                if self.led and self.led.enabled:
                    status_text += " | LED OK"
                else:
                    status_text += " | LED (uitgeschakeld)"

                self.set_status(status_text, COLOR_SUCCESS)
                self.update_ui_state(enabled=True)
                self.prediction_var.set("Klaar")
                if self.led:
                    self.led.send_command("idle")
            elif message_type == "init_error":
                self.initialized = False
                short_error = str(payload).strip().splitlines()[0]
                self.set_status(f"Init Error: {short_error}", COLOR_ERROR)
                self.auto_init_retry_count += 1
                # Probeer beperkt automatisch opnieuw; daarna manueel om een oneindige loop te vermijden.
                if self.auto_init_retry_count <= self.max_auto_init_retries:
                    self.root.after(self.init_retry_delay_ms, self._retry_initialization_if_needed)
                else:
                    self.set_status(
                        "Init blijft falen. Druk op Reset om opnieuw te proberen.",
                        COLOR_ERROR,
                    )
            elif message_type == "init_finished":
                self.init_in_progress = False
            elif message_type == "done":
                self.worker_active = False
                self.update_ui_state(self.initialized)

        self.root.after(50, self._process_worker_messages)

    def _retry_initialization_if_needed(self) -> None:
        if not self.running or self.initialized or self.init_in_progress:
            return
        self.set_status("Init opnieuw proberen...", COLOR_ACCENT)
        self.start_initialization()

    def _show_results(self, probabilities, predicted_idx, inference_time):
        led_cmd = None
        prob = 0.0

        if 0 <= predicted_idx < len(self.classes):
            name = self.classes[predicted_idx]
            color = self.colors[predicted_idx]
            if 0 <= predicted_idx < len(probabilities):
                prob = float(probabilities[predicted_idx])
        else:
            name = self.default_fallback
            color = CLASS_COLOR_MAP.get(name, COLOR_TEXT)

        # Veiligheidsfallback: lage confidence op batterijen -> restafval.
        if name.lower() == "overige/batterijen" and prob < 0.85:
            name = "Restafval"
            color = CLASS_COLOR_MAP.get(name, COLOR_TEXT)
            prob = max(prob, 0.85)

        led_cmd = self._label_to_led_cmd(name)

        # LED-strip aansturen via lokale controller
        if self.led and led_cmd:
            response = self.led.send_command(led_cmd)
            print(f"[LED] Output: {name} ({prob*100:.1f}%) -> Command: {led_cmd}")

        # Update de User Interface (UI)
        self.prediction_var.set(name.upper())
        self.prediction_label.config(fg=color)
        self.confidence_var.set(f"{prob*100:.1f}% zekerheid")

        status_text = f"Inferentie: {inference_time:.1f}ms"
        if led_cmd:
            status_text += f" | LED: {led_cmd}"
        self.set_status(status_text, "#888888")

    @staticmethod
    def _label_to_led_cmd(label: str) -> str:
        normalized = label.strip().lower()
        if normalized.startswith("overige/"):
            return "reject"

        root = normalized.split("/", 1)[0].strip()
        if root in {"organisch", "bio"}:
            return "select_organisch"
        if root == "pmd":
            return "select_pmd"
        if root in {"papier", "karton"}:
            return "select_karton"
        if root == "overige":
            return "reject"
        if root == "restafval":
            return "select_rest"
        return "idle"

    def reset_classification(self) -> None:
        """Reset de classificatie en stop eventuele actieve analyses."""
        # Stop de worker
        self.worker_active = False
        
        # Reset de display
        self.prediction_var.set("Gereed")
        self.prediction_label.config(fg=COLOR_ACCENT)
        self.confidence_var.set("-- %")
        
        # Reset ledstrips
        if self.led:
            self.led.send_command("idle")
        self.set_status("Gereset", COLOR_SUCCESS)

        # Als init nog niet rond is, laat Reset een handmatige herstart van init doen.
        if not self.initialized and not self.init_in_progress:
            self.auto_init_retry_count = 0
            self.set_status("Init opnieuw starten...", COLOR_ACCENT)
            self.start_initialization()
            return
        
        # Heractiveer de UI
        if self.initialized:
            self.update_ui_state(enabled=True)

    def update_ui_state(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        bg_color = COLOR_ACCENT if enabled else "#555555"
        self.btn_classify.config(state=state, bg=bg_color)
        # Reset button blijft altijd enabled als systeem geïnitialiseerd is
        if self.initialized:
            self.btn_reset.config(state="normal")

    def set_status(self, text: str, color: str) -> None:
        self.status_var.set(text)
        # self.status_bar.config(fg=color) # Optioneel

    def toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode on/off."""
        current = self.root.attributes("-fullscreen")
        self.root.attributes("-fullscreen", not current)

    def on_closing(self) -> None:
        self.running = False
        try:
            if self.camera is not None:
                self.camera.stop()
            if self.led is not None:
                self.led.close()
        finally:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_path", help="Specifiek model pad")
    parser.add_argument("--fullscreen", action="store_true", help="Start fullscreen")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    config = DisplayConfig(model_path=args.model_path, fullscreen=args.fullscreen)
    app = InferenceGUI(config)
    app.run()

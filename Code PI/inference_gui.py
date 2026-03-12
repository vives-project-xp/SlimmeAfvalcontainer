import argparse
import threading
import time
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


DEFAULT_CLASSES = ("Organisch", "PMD", "Papier", "Restafval")
DEFAULT_COLORS = ("#4CAF50", "#FFC107", "#2196F3", "#757575")

# Kleurenpalet (Dark Theme)
COLOR_BG = "#1E1E1E"
COLOR_SIDEBAR = "#2D2D2D"
COLOR_TEXT = "#E0E0E0"
COLOR_ACCENT = "#3498DB"
COLOR_SUCCESS = "#2ECC71"
COLOR_ERROR = "#E74C3C"


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

    # Prioriteit 1: inference_model.onnx
    candidates.append((script_dir / "inference_model.onnx").resolve())
    
    # Zoek in AI submap
    ai_subdir = script_dir / "AI"
    if ai_subdir.exists():
        candidates.append((ai_subdir / "inference_model.onnx").resolve())
        candidates.append((ai_subdir / "model.onnx").resolve())

    # Zoek in Ai-model map
    ai_dir = script_dir.parent / "Ai-model"
    if ai_dir.exists():
        candidates.append((ai_dir / "inference_model.onnx").resolve())
        candidates.append((ai_dir / "model.onnx").resolve())

    # Fallback
    candidates.append((script_dir / "model.onnx").resolve())

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



class InferenceGUI:
    def __init__(self, config: DisplayConfig):
        self.config = config
        self.classes = list(DEFAULT_CLASSES)
        self.colors = list(DEFAULT_COLORS)
        self.running = True
        self.initialized = False
        self.worker_active = False
        self.result_queue: Queue[tuple[str, object]] = Queue()
        self.latest_frame: np.ndarray | None = None
        self.session: ort.InferenceSession | None = None
        self.input_name: str | None = None
        self.camera: Picamera2 | None = None
        self.led: LedController | None = None

        self.setup_ui()
        self.update_ui_state(enabled=False)
        self.set_status("Systeem opstarten...", COLOR_ACCENT)
        self.update_preview()
        
        # Start initialisatie in achtergrond
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

        self.btn_reset = tk.Button(
            btn_frame,
            text="RESET",
            command=self.reset_classification,
            font=("Helvetica", 14, "bold"),
            bg=COLOR_ERROR,
            fg="white",
            activebackground="#C0392B",
            activeforeground="white",
            bd=0,
            padx=10,
            pady=15,
            cursor="hand2"
        )
        self.btn_reset.pack(fill=tk.X, pady=10)

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
            candidates = resolve_model_path(self.config.model_path)
            
            session = None
            resolved_model = None
            last_error = None
            input_name = None
            input_shape = None

            for model_path in candidates:
                try:
                    print(f"Trying to load model: {model_path}")
                    # Probeer sessie te maken
                    sess = ort.InferenceSession(model_path)
                    # Check input naam (test of model geldig is)
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
                raise last_error or RuntimeError("Geen geldig ONNX-model gevonden/geladen.")

            print("Initializing camera...")
            camera = Picamera2()
            # Gebruik maximale resolutie voor betere kwaliteit, resize voor preview
            camera_config = camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            camera.configure(camera_config)
            camera.start()
            time.sleep(1)

            print("Initializing LED controller...")
            led = LedController()

            if not self.running:
                camera.stop()
                led.close()
                return
            
            # Detecteer output names (nodig voor object detection)
            output_names = [o.name for o in session.get_outputs()]

            self.result_queue.put(("init_ok", (session, input_name, input_shape, output_names, camera, resolved_model, led)))
        except Exception as exc:
            self.result_queue.put(("init_error", str(exc)))

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

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Default target
        target_h, target_w = 224, 224
        
        # Try to detect dynamic shape from model metadata
        if hasattr(self, 'input_shape') and self.input_shape:
            try:
                # Typically [batch, channels, height, width] or [batch, height, width, channels]
                shape = self.input_shape
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


    def _classify_worker(self) -> None:
        try:
            if self.session is None or self.input_name is None:
                raise RuntimeError("Model niet geladen")

            if self.latest_frame is not None:
                image = self.latest_frame.copy()
            elif self.camera is not None:
                image = self.camera.capture_array()
            else:
                raise RuntimeError("Geen beeld")

            img_array = self.preprocess_image(image)
            start = time.time()
            
            # --- MODEL INFERENCE ---
            inference_outputs = self.session.run(None, {self.input_name: img_array})
            inference_time = (time.time() - start) * 1000

            # --- POST PROCESSING ---
            # Detecteer model type op basis van output namen
            output_names = getattr(self, 'output_names', [])
            
            # Detectie of deze specifieke output names bestaan
            is_detection_model = False
            for name in output_names:
                 if 'dets' in name or 'labels' in name:
                      is_detection_model = True
                      break
            
            if is_detection_model:
                # OBJECT DETECTIE LOGICA (inference_model.onnx)
                # dets: [1, 300, 4] -> Box coordinaten
                # labels: [1, 300, 5] -> Scores per class?
                
                try:
                    # Zoek de juiste indices in de output lijst
                    dets_idx = -1
                    labels_idx = -1
                    for i, name in enumerate(output_names):
                        if 'dets' in name: dets_idx = i
                        if 'labels' in name: labels_idx = i
                    
                    if dets_idx == -1 or labels_idx == -1:
                         # Fallback als we names niet vinden maar wel dachten detection te zijn
                         dets_idx = 0
                         labels_idx = 1

                    # Haal de arrays op
                    # detection output is vaak [batch, num_boxes, coords]
                    # label output is vaak [batch, num_boxes, num_classes]
                    
                    labels_tensor = inference_outputs[labels_idx] # [1, 300, 5]
                    scores_matrix = labels_tensor[0] # [300, 5]
                    
                    # Vind de hoogste score in de hele matrix
                    max_idx_flat = np.argmax(scores_matrix)
                    det_idx, class_idx = np.unravel_index(max_idx_flat, scores_matrix.shape)
                    max_score = float(scores_matrix[det_idx, class_idx])
                    
                    # Maak een dummy probabilities array voor de UI
                    probabilities = np.zeros(len(self.classes) + 2, dtype=np.float32)
                    
                    # Mapping: dit is vaak trial & error.
                    # Aanname: class_idx 0 = eerste default class? Of background?
                    # Meestal is 0 = background in ONNX, dus 1 = Organisch?
                    mapped_idx = class_idx 
                    
                    # Als de score erg laag is, is het misschien niks
                    if max_score < 0.4:
                        mapped_idx = 999 # "Geen object"
                    
                    if mapped_idx < len(probabilities):
                         probabilities[mapped_idx] = max_score
                    
                    predicted_idx = mapped_idx
                except Exception as e:
                    print(f"Error in detection parsing: {e}")
                    probabilities = np.array([0.0] * 4)
                    predicted_idx = 999

            else:
                # STANDAARD CLASSIFICATIE LOGICA (model.onnx)
                # Output[0]: [1, 4]
                raw_output = np.asarray(inference_outputs[0], dtype=np.float32)
                if raw_output.ndim > 1:
                     raw_output = raw_output.flatten() # [4]
                
                # Check of het logits zijn of al probabilities
                if np.max(np.abs(raw_output)) > 1.5:
                     # Waarschijnlijk logits -> softmax
                     exp_x = np.exp(raw_output - np.max(raw_output))
                     probabilities = exp_x / exp_x.sum()
                else:
                     probabilities = raw_output

                predicted_idx = int(np.argmax(probabilities))

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
                session, input_name, input_shape, output_names, camera, resolved_model, led = payload
                self.session = session
                self.input_name = input_name
                self.input_shape = input_shape
                self.output_names = output_names
                self.camera = camera
                self.led = led
                self.initialized = True

                status_text = f"Klaar. Model: {Path(resolved_model).name}"
                if self.led and self.led.enabled:
                    status_text += " | LED OK"
                else:
                    status_text += " | LED (uitgeschakeld)"

                self.set_status(status_text, COLOR_SUCCESS)
                self.update_ui_state(enabled=True)
                self.prediction_var.set("Klaar")
            elif message_type == "init_error":
                self.initialized = False
                self.set_status(f"Init Error: {payload}", COLOR_ERROR)
            elif message_type == "done":
                self.worker_active = False
                self.update_ui_state(self.initialized)

        self.root.after(50, self._process_worker_messages)

    def _show_results(self, probabilities, predicted_idx, inference_time):
        if predicted_idx < len(self.classes):
            name = self.classes[predicted_idx]
            color = self.colors[predicted_idx]
            
            # LED-strip aansturen via lokale controller
            if self.led:
                # ("Organisch", "PMD", "Papier", "Restafval")
                # 0->organisch, 1->pmd, 2->karton, 3->rest
                cmds = ["organisch", "pmd", "karton", "rest"]
                if 0 <= predicted_idx < len(cmds):
                    cmd = cmds[predicted_idx]
                    response = self.led.send_command(cmd)
                    print(f"[LED] {response}")
                    self.set_status(f"LED: {response}", COLOR_SUCCESS)
        else:
            name = f"Onbekend ({predicted_idx})"
            color = COLOR_TEXT

        prob = float(probabilities[predicted_idx]) * 100.0
        
        self.prediction_var.set(name.upper())
        self.prediction_label.config(fg=color) # Update kleur op basis van klasse
        self.confidence_var.set(f"{prob:.1f}% zekerheid")
        self.set_status(f"Inferentie tijd: {inference_time:.1f}ms", "#888888")

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
            self.led.send_command("reset")
        self.set_status("Gereset", COLOR_SUCCESS)
        
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

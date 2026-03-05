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
    from PIL import Image, ImageTk
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
    window_width: int = 1024
    window_height: int = 600
    preview_width: int = 640
    preview_height: int = 480
    fullscreen: bool = False
    rotate: int = 0
    update_ms: int = 50


def resolve_model_path(model_path: str | None = None) -> str:
    """Zoek een bruikbaar ONNX-model op logische locaties."""
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    if model_path:
        user_path = Path(model_path).expanduser()
        if user_path.is_absolute():
            candidates.append(user_path)
        else:
            candidates.append((Path.cwd() / user_path).resolve())
            candidates.append((script_dir / user_path).resolve())

    candidates.append((script_dir / "model.onnx").resolve())

    # Zoek in de AI map (Code PI/AI)
    ai_subdir = script_dir / "AI"
    if ai_subdir.exists():
        candidates.append((ai_subdir / "model.onnx").resolve())

    # Zoek ook in de Ai-model map (Fallback)
    ai_dir = script_dir.parent / "Ai-model"
    if ai_dir.exists():
        candidates.append((ai_dir / "model.onnx").resolve())

    checked: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        checked.append(candidate)
        if candidate.is_file():
            return str(candidate)

    available_models = sorted(path.name for path in script_dir.glob("*.onnx"))
    available_text = ", ".join(available_models) if available_models else "geen"
    checked_text = ", ".join(str(path) for path in checked) if checked else "geen paden"
    raise FileNotFoundError(
        "Geen ONNX-model gevonden.\n"
        f"Geprobeerd: {checked_text}\n"
        f"Beschikbare .onnx-bestanden in {script_dir}: {available_text}"
    )


class SmartBinDisplayApp:
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

        self.setup_gui()
        self._set_buttons_enabled(False)
        self._set_status("Initialiseren: model en camera laden...")
        self.update_preview()
        init_thread = threading.Thread(target=self._initialize_worker)
        init_thread.daemon = True
        init_thread.start()

    def setup_gui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Smart Bin Display")
        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")
        self.root.configure(bg=COLOR_BG)
        self.root.resizable(True, True)

        if self.config.fullscreen:
            self.root.attributes("-fullscreen", True)

        self.root.bind("<Escape>", lambda _event: self.on_closing())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Styling
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            "Classify.Horizontal.TProgressbar",
            thickness=15,
            troughcolor=COLOR_SIDEBAR,
            background=COLOR_ACCENT,
            borderwidth=0
        )

        # Main Container
        main_container = tk.Frame(self.root, bg=COLOR_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left Column (Camera)
        left_col = tk.Frame(main_container, bg="black", width=self.config.preview_width, height=self.config.preview_height)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_col.pack_propagate(False)

        self.preview_label = tk.Label(
            left_col,
            bg="black",
            fg="white",
            text="Camera wordt gestart...",
            font=("Helvetica", 16),
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # Right Column (Sidebar)
        right_col = tk.Frame(main_container, bg=COLOR_SIDEBAR, width=320)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_col.pack_propagate(False)

        # Header Title
        tk.Label(
            right_col,
            text="Slimme\nAfvalcontainer",
            font=("Helvetica", 20, "bold"),
            bg=COLOR_SIDEBAR,
            fg="white",
            justify="center",
        ).pack(pady=(20, 20))

        # Prediction Result (Big)
        self.prediction_var = tk.StringVar(value="Gereed")
        self.prediction_label = tk.Label(
            right_col,
            textvariable=self.prediction_var,
            font=("Helvetica", 24, "bold"),
            bg=COLOR_SIDEBAR,
            fg=COLOR_ACCENT,
            wraplength=280,
            justify="center"
        )
        self.prediction_label.pack(pady=(0, 5))

        self.time_label = tk.Label(
            right_col,
            text="",
            font=("Helvetica", 10),
            bg=COLOR_SIDEBAR,
            fg="#AAAAAA",
        )
        self.time_label.pack(fill=tk.X, pady=(0, 15))

        # Progress Bars Container
        stats_frame = tk.Frame(right_col, bg=COLOR_SIDEBAR)
        stats_frame.pack(fill=tk.X, padx=15, pady=5)

        self.progress_bars: dict[str, dict[str, object]] = {}
        for class_name in self.classes:
            row = tk.Frame(stats_frame, bg=COLOR_SIDEBAR)
            row.pack(fill=tk.X, pady=4)

            # Label + Percentage on top line
            info_row = tk.Frame(row, bg=COLOR_SIDEBAR)
            info_row.pack(fill=tk.X)
            
            tk.Label(
                info_row,
                text=class_name,
                font=("Helvetica", 10),
                bg=COLOR_SIDEBAR,
                fg=COLOR_TEXT,
                anchor="w"
            ).pack(side=tk.LEFT)

            percentage = tk.Label(
                info_row,
                text="0%",
                font=("Helvetica", 10, "bold"),
                bg=COLOR_SIDEBAR,
                fg="#888888",
                anchor="e"
            )
            percentage.pack(side=tk.RIGHT)

            # Bar
            progress = ttk.Progressbar(
                row,
                length=100,
                mode="determinate",
                maximum=100,
                style="Classify.Horizontal.TProgressbar",
            )
            progress.pack(fill=tk.X, pady=(2, 0))

            self.progress_bars[class_name] = {"bar": progress, "label": percentage}

        # Buttons
        button_frame = tk.Frame(right_col, bg=COLOR_SIDEBAR)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

        self.classify_btn = tk.Button(
            button_frame,
            text="ANALYSEER NU",
            command=self.classify_threaded,
            font=("Helvetica", 12, "bold"),
            bg=COLOR_ACCENT,
            fg="white",
            activebackground="#2980B9",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=12
        )
        self.classify_btn.pack(fill=tk.X, pady=(0, 10))

        self.save_btn = tk.Button(
            button_frame,
            text="OPSLAAN",
            command=lambda: self.classify_threaded(save=True),
            font=("Helvetica", 10, "bold"),
            bg="#555555",
            fg="white",
            activebackground="#777777",
            activeforeground="white",
            relief=tk.FLAT,
            cursor="hand2",
            padx=10,
            pady=8
        )
        self.save_btn.pack(fill=tk.X)

        # Status Bar
        self.status_label = tk.Label(
            right_col,
            text="Initialiseren...",
            font=("Helvetica", 9),
            bg=COLOR_SIDEBAR,
            fg="#666666",
            anchor="w",
            padx=10
        )
        # We pack status label inside right col at bottom, above buttons? 
        # Actually putting it at very bottom of Sidebar looks cleaner
        # Let's repack buttons to be slightly higher or status at very bottom
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 5))

        self.root.after(50, self._process_worker_messages)

    def _initialize_worker(self) -> None:
        try:
            print("Initializing model...")
            resolved_model = resolve_model_path(self.config.model_path)
            print(f"Loading model: {resolved_model}")
            session = ort.InferenceSession(resolved_model)
            input_name = session.get_inputs()[0].name

            print("Initializing camera...")
            camera = Picamera2()
            camera_config = camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            camera.configure(camera_config)
            camera.start()
            time.sleep(1)

            if not self.running:
                camera.stop()
                return

            self.result_queue.put(("init_ok", (session, input_name, camera, resolved_model)))
        except Exception as exc:  # noqa: BLE001
            self.result_queue.put(("init_error", str(exc)))

    def update_preview(self) -> None:
        if not self.running:
            return

        if self.camera is None:
            self.root.after(self.config.update_ms, self.update_preview)
            return

        try:
            image = self.camera.capture_array()
        except Exception as exc:  # noqa: BLE001
            self._set_error(f"Camerafout: {exc}")
            self._set_status("Camera kon niet lezen. Controleer camera-aansluiting.", "#B00020")
            self.root.after(self.config.update_ms, self.update_preview)
            return

        # Picamera2 levert BGR, PIL verwacht RGB – draai kanalen om
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
        self.preview_label.image = photo
        self.root.after(self.config.update_ms, self.update_preview)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        img = Image.fromarray(image).resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        return img_array

    def classify_threaded(self, save: bool = False) -> None:
        if not self.running or self.worker_active:
            return
        if not self.initialized:
            self._set_error("Nog niet klaar: model/camera initialiseren.")
            return

        self.worker_active = True
        self._set_buttons_enabled(False)
        thread = threading.Thread(target=self._classify_worker, args=(save,))
        thread.daemon = True
        thread.start()

    def _classify_worker(self, save: bool) -> None:
        try:
            self.classify(save_photo=save)
        except Exception as exc:  # noqa: BLE001
            self.result_queue.put(("error", str(exc)))
        finally:
            self.result_queue.put(("done", None))

    def classify(self, save_photo: bool = False) -> None:
        if self.session is None or self.input_name is None:
            raise RuntimeError("Model is nog niet geladen.")

        if self.latest_frame is not None:
            image = self.latest_frame.copy()
        else:
            if self.camera is None:
                raise RuntimeError("Camera is nog niet klaar.")
            image = self.camera.capture_array()

        if save_photo:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = Path(__file__).resolve().parent / f"capture_{timestamp}.jpg"
            Image.fromarray(image).save(output_path)
            print(f"Saved: {output_path}")

        img_array = self.preprocess_image(image)
        start = time.time()
        outputs = self.session.run(None, {self.input_name: img_array})
        inference_time = (time.time() - start) * 1000

        raw_output = np.asarray(outputs[0][0], dtype=np.float32)
        # Apply softmax explicitly to ensure we have probabilities (0-1)
        # This fixes issues where the model outputs logits (unbounded numbers)
        exp_x = np.exp(raw_output - np.max(raw_output))
        probabilities = exp_x / exp_x.sum()

        predicted_idx = int(np.argmax(probabilities))
        self.result_queue.put(("result", (probabilities, predicted_idx, inference_time)))

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
                self._update_results(probabilities, predicted_idx, inference_time)
            elif message_type == "error":
                self._set_error(str(payload))
            elif message_type == "init_ok":
                session, input_name, camera, resolved_model = payload
                self.session = session
                self.input_name = input_name
                self.camera = camera
                self.initialized = True
                self.prediction_label.config(text="Klaar voor classificatie", fg="#333333")
                self._set_status(f"Gereed. Model: {Path(resolved_model).name}")
                self._set_buttons_enabled(True)
            elif message_type == "init_error":
                self.initialized = False
                self._set_error(str(payload))
                self._set_status("Initialisatie mislukt. Check terminal-output.", "#B00020")
                self._set_buttons_enabled(False)
            elif message_type == "done":
                self.worker_active = False
                self._set_buttons_enabled(self.initialized)

        self.root.after(50, self._process_worker_messages)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        bg_color = COLOR_ACCENT if enabled else "#555555"
        self.classify_btn.config(state=state, bg=bg_color)
        self.save_btn.config(state=state)

    def _set_error(self, message: str) -> None:
        self.prediction_label.config(text="Fout", fg=COLOR_ERROR)
        self.status_label.config(text=f"Fout: {message}", fg=COLOR_ERROR)

    def _set_status(self, message: str, color: str = "#666666") -> None:
        self.status_label.config(text=message, fg=color)

    def _update_results(
        self,
        probabilities: np.ndarray,
        predicted_idx: int,
        inference_time: float,
    ) -> None:
        if predicted_idx < len(self.classes):
            predicted_name = self.classes[predicted_idx]
            predicted_color = self.colors[predicted_idx]
        else:
            predicted_name = f"Klasse {predicted_idx}"
            predicted_color = COLOR_TEXT

        self.prediction_label.config(
            text=f"{predicted_name}",
            fg=predicted_color,
        )
        self.time_label.config(text=f"Inferentie: {inference_time:.1f} ms")

        # Update Progress Bars
        for index, class_name in enumerate(self.classes):
            probability = float(probabilities[index]) * 100.0 if index < len(probabilities) else 0.0
            bar = self.progress_bars[class_name]["bar"]
            label = self.progress_bars[class_name]["label"]
            
            bar["value"] = probability
            label.config(text=f"{probability:.1f}%")
            
            # Dynamic color for bar? (Not easy with standard ttk theme on linux without heavy styling)
            # We keep standard accent color.

        predicted_prob = (
            float(probabilities[predicted_idx]) * 100.0
            if predicted_idx < len(probabilities)
            else 0.0
        )
        # Also print to terminal
        print(
            f"Voorspelling: {predicted_name} "
            f"({predicted_prob:.1f}%)"
        )

    def on_closing(self) -> None:
        self.running = False
        try:
            if self.camera is not None:
                self.camera.stop()
        finally:
            self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Bin Display GUI")
    parser.add_argument(
        "--model",
        dest="model_path",
        default=None,
        help="Pad naar ONNX-modelbestand (bijv. model.onnx)",
    )
    parser.add_argument("--width", type=int, default=800, help="Display breedte in pixels")
    parser.add_argument("--height", type=int, default=600, help="Display hoogte in pixels")
    parser.add_argument(
        "--preview-width",
        type=int,
        default=400,
        help="Breedte van camera preview",
    )
    parser.add_argument(
        "--preview-height",
        type=int,
        default=300,
        help="Hoogte van camera preview",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
        help="Rotatie van camerabeeld voor je display",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        default=False,
        help="Start in fullscreen mode",
    )
    parser.add_argument(
        "--no-fullscreen",
        dest="fullscreen",
        action="store_false",
        help="Schakel fullscreen uit (bijv. voor desktop gebruik)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = DisplayConfig(
        model_path=args.model_path,
        window_width=max(args.width, 240),
        window_height=max(args.height, 160),
        preview_width=max(args.preview_width, 120),
        preview_height=max(args.preview_height, 90),
        rotate=args.rotate,
        fullscreen=args.fullscreen,
    )

    try:
        print("Starting Smart Bin Display GUI...")
        app = SmartBinDisplayApp(config)
        app.run()
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

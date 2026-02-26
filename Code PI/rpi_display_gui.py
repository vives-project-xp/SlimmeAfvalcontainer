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


@dataclass(frozen=True)
class DisplayConfig:
    model_path: str | None = None
    window_width: int = 480
    window_height: int = 320
    preview_width: int = 300
    preview_height: int = 225
    fullscreen: bool = False
    rotate: int = 0
    update_ms: int = 100


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
    candidates.append((script_dir / "modelv2.onnx").resolve())

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
        self.root.configure(bg="#f2f2f2")
        self.root.resizable(False, False)

        if self.config.fullscreen:
            self.root.attributes("-fullscreen", True)

        self.root.bind("<Escape>", lambda _event: self.on_closing())

        self.style = ttk.Style()
        self.style.configure("Classify.Horizontal.TProgressbar", thickness=12)

        header = tk.Label(
            self.root,
            text="Slimme Vuilnisbak",
            font=("Arial", 14, "bold"),
            bg="#f2f2f2",
            fg="#222222",
        )
        header.pack(pady=(6, 2))

        content = tk.Frame(self.root, bg="#f2f2f2")
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        preview_frame = tk.Frame(
            content,
            bg="black",
            width=self.config.preview_width,
            height=self.config.preview_height,
        )
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        preview_frame.pack_propagate(False)

        self.preview_label = tk.Label(
            preview_frame,
            bg="black",
            fg="white",
            text="Camera wordt gestart...",
            font=("Arial", 10),
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        side_panel = tk.Frame(content, bg="#f2f2f2")
        side_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))

        self.prediction_label = tk.Label(
            side_panel,
            text="Wacht op classificatie...",
            font=("Arial", 12, "bold"),
            bg="#f2f2f2",
            fg="#333333",
            justify="left",
            anchor="w",
            wraplength=max(self.config.window_width - self.config.preview_width - 40, 120),
        )
        self.prediction_label.pack(fill=tk.X, pady=(0, 4))

        self.time_label = tk.Label(
            side_panel,
            text="",
            font=("Arial", 10),
            bg="#f2f2f2",
            fg="#666666",
            anchor="w",
        )
        self.time_label.pack(fill=tk.X, pady=(0, 6))

        self.status_label = tk.Label(
            side_panel,
            text="",
            font=("Arial", 9),
            bg="#f2f2f2",
            fg="#555555",
            anchor="w",
            justify="left",
            wraplength=max(self.config.window_width - self.config.preview_width - 40, 120),
        )
        self.status_label.pack(fill=tk.X, pady=(0, 6))

        self.progress_bars: dict[str, dict[str, object]] = {}
        for class_name in self.classes:
            row = tk.Frame(side_panel, bg="#f2f2f2")
            row.pack(fill=tk.X, pady=2)

            label = tk.Label(
                row,
                text=f"{class_name}:",
                font=("Arial", 9),
                width=9,
                anchor="w",
                bg="#f2f2f2",
            )
            label.pack(side=tk.LEFT)

            progress = ttk.Progressbar(
                row,
                length=110,
                mode="determinate",
                maximum=100,
                style="Classify.Horizontal.TProgressbar",
            )
            progress.pack(side=tk.LEFT, padx=(4, 6))

            percentage = tk.Label(
                row,
                text="0%",
                font=("Arial", 9),
                width=5,
                anchor="e",
                bg="#f2f2f2",
            )
            percentage.pack(side=tk.LEFT)

            self.progress_bars[class_name] = {"bar": progress, "label": percentage}

        button_frame = tk.Frame(side_panel, bg="#f2f2f2")
        button_frame.pack(fill=tk.X, pady=(8, 0))

        self.classify_btn = tk.Button(
            button_frame,
            text="Classificeer",
            command=self.classify_threaded,
            font=("Arial", 10, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=8,
            pady=6,
        )
        self.classify_btn.pack(fill=tk.X, pady=(0, 4))

        self.save_btn = tk.Button(
            button_frame,
            text="Classificeer + Opslaan",
            command=lambda: self.classify_threaded(save=True),
            font=("Arial", 10, "bold"),
            bg="#2196F3",
            fg="white",
            padx=8,
            pady=6,
        )
        self.save_btn.pack(fill=tk.X)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
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

        self.latest_frame = image.copy()
        img = Image.fromarray(image)

        if self.config.rotate:
            img = img.rotate(self.config.rotate, expand=True)

        img = img.resize(
            (self.config.preview_width, self.config.preview_height),
            Image.Resampling.LANCZOS,
        )
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

        probabilities = np.asarray(outputs[0][0], dtype=np.float32)
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
        self.classify_btn.config(state=state)
        self.save_btn.config(state=state)

    def _set_error(self, message: str) -> None:
        self.prediction_label.config(text=f"Fout: {message}", fg="#B00020")

    def _set_status(self, message: str, color: str = "#555555") -> None:
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
            predicted_color = "#333333"

        self.prediction_label.config(
            text=f"Voorspelling: {predicted_name}",
            fg=predicted_color,
        )
        self.time_label.config(text=f"Inferentie: {inference_time:.1f} ms")

        for index, class_name in enumerate(self.classes):
            probability = float(probabilities[index]) * 100.0 if index < len(probabilities) else 0.0
            bar = self.progress_bars[class_name]["bar"]
            label = self.progress_bars[class_name]["label"]
            bar["value"] = probability
            label.config(text=f"{probability:.1f}%")

        predicted_prob = (
            float(probabilities[predicted_idx]) * 100.0
            if predicted_idx < len(probabilities)
            else 0.0
        )
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
    parser = argparse.ArgumentParser(description="Smart Bin 3.5-inch Display GUI")
    parser.add_argument(
        "--model",
        dest="model_path",
        default=None,
        help="Pad naar ONNX-modelbestand (bijv. model.onnx)",
    )
    parser.add_argument("--width", type=int, default=480, help="Display breedte in pixels")
    parser.add_argument("--height", type=int, default=320, help="Display hoogte in pixels")
    parser.add_argument(
        "--preview-width",
        type=int,
        default=300,
        help="Breedte van camera preview",
    )
    parser.add_argument(
        "--preview-height",
        type=int,
        default=225,
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
        help="Start in fullscreen mode (handig op dedicated Pi display)",
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

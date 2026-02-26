import onnxruntime as ort
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
import threading
import argparse
from pathlib import Path

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

def resolve_model_path(model_path=None):
    """Zoek een bruikbaar ONNX-model op een paar logische locaties."""
    script_dir = Path(__file__).resolve().parent
    candidates = []

    if model_path:
        user_path = Path(model_path).expanduser()
        if user_path.is_absolute():
            candidates.append(user_path)
        else:
            candidates.append((Path.cwd() / user_path).resolve())
            candidates.append((script_dir / user_path).resolve())

    candidates.append((script_dir / "model.onnx").resolve())
    candidates.append((script_dir / "modelv2.onnx").resolve())

    checked = []
    seen = set()
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

class SmartBinClassifierGUI:
    def __init__(self, model_path=None):
        """Initialiseer camera, model en GUI"""
        print("Initializing model...")
        resolved_model = resolve_model_path(model_path)
        print(f"Loading model: {resolved_model}")
        self.session = ort.InferenceSession(resolved_model)
        self.classes = ['Organisch', 'PMD', 'Papier', 'Restafval']
        self.colors = ['#4CAF50', '#FFC107', '#2196F3', '#757575']  # Groen, Geel, Blauw, Grijs
        
        print("Initializing camera...")
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(2)
        
        # GUI setup
        self.setup_gui()
        self.running = True
        self.update_preview()
        
    def setup_gui(self):
        """Maak GUI venster"""
        self.root = tk.Tk()
        self.root.title("Smart Bin Classifier")
        self.root.geometry("800x700")
        self.root.configure(bg='#f0f0f0')
        
        # Header
        header = tk.Label(
            self.root, 
            text="Smart Vuilnisbak Classifier", 
            font=('Arial', 20, 'bold'),
            bg='#f0f0f0'
        )
        header.pack(pady=10)
        
        # Camera preview
        self.preview_label = tk.Label(self.root, bg='black')
        self.preview_label.pack(pady=10)
        
        # Buttons frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        self.classify_btn = tk.Button(
            button_frame,
            text="📸 Classificeer",
            command=self.classify_threaded,
            font=('Arial', 14, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10
        )
        self.classify_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = tk.Button(
            button_frame,
            text="💾 Classificeer & Opslaan",
            command=lambda: self.classify_threaded(save=True),
            font=('Arial', 14, 'bold'),
            bg='#2196F3',
            fg='white',
            padx=20,
            pady=10
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = tk.Frame(self.root, bg='#f0f0f0')
        results_frame.pack(pady=20, fill=tk.BOTH, expand=True, padx=20)
        
        # Prediction label
        self.prediction_label = tk.Label(
            results_frame,
            text="Wacht op classificatie...",
            font=('Arial', 18, 'bold'),
            bg='#f0f0f0',
            fg='#333'
        )
        self.prediction_label.pack(pady=10)
        
        # Inference time
        self.time_label = tk.Label(
            results_frame,
            text="",
            font=('Arial', 12),
            bg='#f0f0f0',
            fg='#666'
        )
        self.time_label.pack()
        
        # Progress bars for all classes
        self.progress_bars = {}
        for i, cls in enumerate(self.classes):
            frame = tk.Frame(results_frame, bg='#f0f0f0')
            frame.pack(fill=tk.X, pady=5)
            
            label = tk.Label(
                frame,
                text=f"{cls}:",
                font=('Arial', 12),
                bg='#f0f0f0',
                width=12,
                anchor='w'
            )
            label.pack(side=tk.LEFT)
            
            progress = ttk.Progressbar(
                frame,
                length=400,
                mode='determinate',
                maximum=100
            )
            progress.pack(side=tk.LEFT, padx=10)
            
            percentage = tk.Label(
                frame,
                text="0%",
                font=('Arial', 12),
                bg='#f0f0f0',
                width=6,
                anchor='e'
            )
            percentage.pack(side=tk.LEFT)
            
            self.progress_bars[cls] = {
                'bar': progress,
                'label': percentage
            }
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def update_preview(self):
        """Update camera preview"""
        if not self.running:
            return
            
        image = self.camera.capture_array()
        
        # Convert RGB to BGR voor correcte kleuren
        image = image[:, :, ::-1]  # Voeg deze regel toe
        
        # Resize voor display
        img = Image.fromarray(image)
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo
        
        # Update elke 100ms
        self.root.after(100, self.update_preview)
    
    def preprocess_image(self, image):
        """Preprocess afbeelding voor model"""
        img = Image.fromarray(image).resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        return img_array
    
    def classify(self, save_photo=False):
        """Classificeer huidige camera beeld"""
        # Disable buttons
        self.classify_btn.config(state='disabled')
        self.save_btn.config(state='disabled')
        
        # Capture image
        image = self.camera.capture_array()
        
        # Save if requested
        if save_photo:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            Image.fromarray(image).save(f"capture_{timestamp}.jpg")
            print(f"Saved: capture_{timestamp}.jpg")
        
        # Preprocess
        img_array = self.preprocess_image(image)
        
        # Inference
        start = time.time()
        outputs = self.session.run(None, {"input": img_array})
        inference_time = (time.time() - start) * 1000
        
        # Results
        probabilities = outputs[0][0]
        predicted_idx = np.argmax(probabilities)
        
        # Update GUI
        self.prediction_label.config(
            text=f"Voorspelling: {self.classes[predicted_idx]}",
            fg=self.colors[predicted_idx]
        )
        self.time_label.config(
            text=f"Inferentie tijd: {inference_time:.2f}ms"
        )
        
        # Update progress bars
        for i, cls in enumerate(self.classes):
            prob = probabilities[i] * 100
            self.progress_bars[cls]['bar']['value'] = prob
            self.progress_bars[cls]['label'].config(text=f"{prob:.1f}%")
        
        # Re-enable buttons
        self.classify_btn.config(state='normal')
        self.save_btn.config(state='normal')
        
        print(f"\nVoorspelling: {self.classes[predicted_idx]} ({probabilities[predicted_idx]*100:.1f}%)")
    
    def classify_threaded(self, save=False):
        """Run classificatie in aparte thread"""
        thread = threading.Thread(target=self.classify, args=(save,))
        thread.daemon = True
        thread.start()
    
    def on_closing(self):
        """Cleanup bij afsluiten"""
        self.running = False
        self.camera.stop()
        self.root.destroy()
    
    def run(self):
        """Start de GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Bin Classifier GUI")
    parser.add_argument(
        "--model",
        dest="model_path",
        default=None,
        help="Pad naar ONNX-modelbestand (bijv. modelv2.onnx)",
    )
    args = parser.parse_args()

    try:
        print("Starting Smart Bin Classifier GUI...")
        classifier = SmartBinClassifierGUI(model_path=args.model_path)
        classifier.run()
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

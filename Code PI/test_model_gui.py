import onnxruntime as ort
import numpy as np
from PIL import Image, ImageTk
from picamera2 import Picamera2
import time
import tkinter as tk
from tkinter import ttk
import threading

class SmartBinClassifierGUI:
    def __init__(self, model_path="model.onnx"):
        """Initialiseer camera, model en GUI"""
        print("Initializing model...")
        self.session = ort.InferenceSession(model_path)
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
    try:
        print("Starting Smart Bin Classifier GUI...")
        classifier = SmartBinClassifierGUI()
        classifier.run()
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

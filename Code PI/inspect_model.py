import onnxruntime as ort
import numpy as np
import sys
import os

def inspect_model(model_path):
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return

    print(f"--- Inspecting {model_path} ---")
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Inputs:")
    for i in session.get_inputs():
        print(f"  Name: {i.name}, Shape: {i.shape}, Type: {i.type}")

    print("\nOutputs:")
    for o in session.get_outputs():
        print(f"  Name: {o.name}, Shape: {o.shape}, Type: {o.type}")

if __name__ == "__main__":
    # Check default locations
    paths = [
        "Code PI/AI/inference_model.onnx",
        "Code PI/inference_model.onnx", 
        "Ai-model/inference_model.onnx",
        "Code PI/model.onnx",
        "Code PI/AI/model.onnx"
    ]
    
    found = False
    for p in paths:
        if os.path.exists(p):
            inspect_model(p)
            found = True
            print("\n")
            
    if not found:
        print("No models found to inspect.")

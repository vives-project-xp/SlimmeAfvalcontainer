import os
import requests
from pathlib import Path

def download_file(url, destination):
    """Download bestand van URL naar lokale destinatie"""
    print(f"Downloading {os.path.basename(destination)}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Maak directory als die niet bestaat
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download met progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✓ Downloaded: {os.path.basename(destination)}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading {os.path.basename(destination)}: {e}")
        return False

def download_model_files():
    """Download model.onnx en model.onnx.data van GitHub"""
    
    # GitHub raw content URLs
    base_url = "https://raw.githubusercontent.com/vives-project-xp/SlimmeAfvalcontainer/main/Ai-model"
    
    files = {
        "model.onnx": f"{base_url}/model.onnx",
        "model.onnx.data": f"{base_url}/model.onnx.data"
    }
    
    print("=" * 50)
    print("Downloading model files from GitHub...")
    print("=" * 50)
    
    success_count = 0
    for filename, url in files.items():
        if download_file(url, filename):
            success_count += 1
    
    print("\n" + "=" * 50)
    if success_count == len(files):
        print("✓ All files downloaded successfully!")
        print(f"Model files saved in: {os.getcwd()}")
    else:
        print(f"⚠ Downloaded {success_count}/{len(files)} files")
    print("=" * 50)

if __name__ == "__main__":
    # Controleer of bestanden al bestaan
    if os.path.exists("model.onnx") and os.path.exists("model.onnx.data"):
        print("Model files already exist!")
        overwrite = input("Do you want to re-download them? (y/n): ").lower()
        if overwrite != 'y':
            print("Skipping download.")
            exit(0)
    
    download_model_files()

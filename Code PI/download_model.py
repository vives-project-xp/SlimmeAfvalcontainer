import shutil
from pathlib import Path

TARGET_DIR = Path("/home/kobe/smart_bin")
SCRIPT_DIR = Path(__file__).resolve().parent

def sync_code_pi_to_target():
    """Kopieer alle inhoud van Code PI naar /home/kobe/smart_bin"""
    print("=" * 50)
    print(f"Syncing files to {TARGET_DIR}...")
    print("=" * 50)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for item in SCRIPT_DIR.iterdir():
        if item.name == "__pycache__":
            continue

        destination = TARGET_DIR / item.name

        if item.resolve() == destination.resolve():
            continue

        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)

    print(f"✓ Code files synced to: {TARGET_DIR}")

    if (TARGET_DIR / "model.onnx").exists() and (TARGET_DIR / "model.onnx.data").exists():
        print("✓ Modelbestanden gevonden in doelmap.")
    else:
        print("⚠ Let op: model.onnx en/of model.onnx.data ontbreken in Code PI.")

if __name__ == "__main__":
    sync_code_pi_to_target()

"""
Train YOLOv8l on the expanded waste dataset (7 categories).
New classes vs the original:
  5 = Elektronica  (WEEE: smartphones, laptops, cables, kleine apparaten)
  6 = Glas         (flessen, potten, glazen)

You need annotated images for the new classes before they can be learned.
Use label-studio or roboflow to annotate, then run this script.
"""
import os
import glob
import yaml
import random
from ultralytics import YOLO

DATASET_DIR = "/root/smart_bin_project/data/Dataset/train"
CLASSES = [
    "Organisch",    # 0 - GFT
    "PMD",          # 1 - Plastic, Metaal, Drankkartons
    "Papier",       # 2 - Papier & Karton
    "Restafval",    # 3 - Niet-recycleerbaar
    "Overige",      # 4 - Textiel, groot huisvuil, ...
    "Elektronica",  # 5 - WEEE: telefoons, laptops, kabels, kleine apparaten
    "Glas",         # 6 - Glazen flessen & potten
]

def build_file_lists():
    txt_files = glob.glob(os.path.join(DATASET_DIR, "**", "*.txt"), recursive=True)

    labeled_images = []
    for txt in txt_files:
        jpg = os.path.splitext(txt)[0] + ".jpg"
        if os.path.exists(jpg):
            with open(txt, "r") as f:
                if f.read().strip():
                    labeled_images.append(jpg)

    print(f"Geannoteerde afbeeldingen gevonden: {len(labeled_images)}")
    if len(labeled_images) < 10:
        print("Te weinig annotaties. Voeg meer toe en probeer opnieuw.")
        return False

    random.seed(42)
    random.shuffle(labeled_images)
    split = int(len(labeled_images) * 0.8)

    with open("/root/smart_bin_project/yolo_train.txt", "w") as f:
        f.write("\n".join(labeled_images[:split]))
    with open("/root/smart_bin_project/yolo_val.txt", "w") as f:
        f.write("\n".join(labeled_images[split:]))

    print(f"  Train: {split}  |  Val: {len(labeled_images) - split}")
    return True


def write_dataset_yaml():
    yaml_path = "/root/smart_bin_project/dataset_extended.yaml"
    content = {
        "path": "/root/smart_bin_project",
        "train": "yolo_train.txt",
        "val": "yolo_val.txt",
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)
    print(f"Dataset YAML opgeslagen: {yaml_path}")
    return yaml_path


def main():
    print("=== YOLOv8l training – uitgebreide afvaldetectie ===\n")

    if not build_file_lists():
        return

    yaml_path = write_dataset_yaml()

    model = YOLO("yolov8l.pt")

    model.train(
        data=yaml_path,
        epochs=80,
        imgsz=640,      # grotere input dan de n-versie → betere small-object detectie
        batch=16,       # RTX 4000 Ada 20 GB heeft hier ruimte voor
        project="yolo_runs",
        name="garbage_detector_l",
        device="0",
        patience=15,    # vroeg stoppen als val-loss stagneert
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    print("\nTraining klaar!")
    print("Best model: yolo_runs/garbage_detector_l/weights/best.pt")
    print("\nExporteer naar ONNX voor CPU-inferentie:")
    print("  from ultralytics import YOLO")
    print("  YOLO('yolo_runs/garbage_detector_l/weights/best.pt').export(format='onnx', imgsz=640)")


if __name__ == "__main__":
    main()

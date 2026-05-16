#!/usr/bin/env python3
"""
Two-stage classifier training using YOLO-detected crops.

Pipeline:
  1. Run YOLO detector over source dataset → save object crops per class
  2. Train Stage-1 MobileNetV3-Large: 5 main classes
     (Organisch, PMD, Papier, Restafval, Overige)
  3. Train Stage-2 MobileNetV3-Large: 5 Overige sub-classes
     (Batterijen, Elektronica, Glas, Lightbulbs, Metaal)

VRAM limit: 18 GB (no swap memory).
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
SOURCE_ROOT = Path("/root/smart_bin_project/data/Dataset/train")
BEST_DETECTOR = Path(
    "/root/smart_bin_project/runs/detect_strong/"
    "garbage_detector_l_fallback_aware_768-6/weights/best.pt"
)
CROPS_ROOT = Path("/root/smart_bin_project/data/classifier_crops")
OUTPUT_DIR = Path("/root/smart_bin_project/models/two_stage_crops")
LOG_PATH = Path("/root/smart_bin_project/two_stage_crops_training.log")

VRAM_LIMIT_MB = 18 * 1024          # 18 GB  → no swap
EPOCHS_STAGE1 = int(os.getenv("EPOCHS_STAGE1", "50"))
EPOCHS_STAGE2 = int(os.getenv("EPOCHS_STAGE2", "50"))
BATCH_SIZE     = int(os.getenv("BATCH_SIZE",    "64"))
NUM_WORKERS    = int(os.getenv("NUM_WORKERS",   "8"))
LR             = float(os.getenv("LEARNING_RATE", "0.001"))
IMG_SIZE       = 224
SEED           = 42
CONF_THRESH    = 0.25
PADDING        = 0.05
VAL_RATIO      = 0.2
USE_FULL_WHEN_MISSING = True   # fallback to full image if no detection

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
_log_fh = None


def log(msg: str):
    global _log_fh
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_fh is not None:
        _log_fh.write(line + "\n")
        _log_fh.flush()


# ──────────────────────────────────────────────────────────────
# GPU / VRAM helpers
# ──────────────────────────────────────────────────────────────
def setup_vram_limit():
    """Hard-cap GPU memory to VRAM_LIMIT_MB, disable swap (expandable_segments=False)."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        f"max_split_size_mb:{VRAM_LIMIT_MB},"
        "expandable_segments:False,"
        "garbage_collection_threshold:0.8"
    )
    if torch.cuda.is_available():
        # Reserve the fraction we want and refuse further growth
        fraction = VRAM_LIMIT_MB / torch.cuda.get_device_properties(0).total_memory * (1024 ** 2)
        fraction = min(fraction, 0.98)
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        log(
            f"VRAM cap: {VRAM_LIMIT_MB} MB  "
            f"({fraction * 100:.1f}% of "
            f"{torch.cuda.get_device_properties(0).total_memory // (1024**2)} MB total)"
        )
        log("Swap / expandable segments: DISABLED")


def gpu_mem_str() -> str:
    if not torch.cuda.is_available():
        return ""
    a = torch.cuda.memory_allocated(0) // (1024 ** 2)
    r = torch.cuda.memory_reserved(0) // (1024 ** 2)
    return f"GPU {a}/{r} MB"


# ──────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────
# Phase A: Build crop dataset via YOLO detector
# ──────────────────────────────────────────────────────────────
def _list_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VALID_EXTS)


def _choose_box(dets, img_w, img_h):
    """Pick highest-confidence detection."""
    if not dets:
        return None
    return max(dets, key=lambda d: float(d[4]))


def _expand_box(x1, y1, x2, y2, img_w, img_h, pad=PADDING):
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pw, ph = int(bw * pad), int(bh * pad)
    x1e = max(0, x1 - pw)
    y1e = max(0, y1 - ph)
    x2e = min(img_w, x2 + pw)
    y2e = min(img_h, y2 + ph)
    if x2e <= x1e or y2e <= y1e:
        return None
    return x1e, y1e, x2e, y2e


def build_crops(crops_root: Path, force_rebuild=False):
    """
    For every image in SOURCE_ROOT/<class>/… run YOLO detection, crop the
    best box and save it to crops_root/<split>/<class>/….
    """
    marker = crops_root / ".build_done"
    if marker.exists() and not force_rebuild:
        log(f"Crop dataset already exists at {crops_root} – skipping rebuild.")
        return

    log("=" * 60)
    log("Phase A: Building YOLO-crop dataset …")
    log(f"  Source : {SOURCE_ROOT}")
    log(f"  Output : {crops_root}")
    log(f"  Detector: {BEST_DETECTOR}")

    # Import ultralytics here so we don't fail early if it's missing
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError as e:
        log(f"FATAL: {e}")
        sys.exit(1)

    detector = YOLO(str(BEST_DETECTOR))
    rng = random.Random(SEED)

    images = _list_images(SOURCE_ROOT)
    log(f"  Total source images: {len(images)}")

    written = detected = missing = unreadable = 0

    for img_path in tqdm(images, desc="Building crops", unit="img", dynamic_ncols=True):
        # rel class label  e.g. "PMD" or "Overige/Elektronica"
        rel_class = img_path.parent.relative_to(SOURCE_ROOT).as_posix()
        if rel_class == ".":
            rel_class = "unknown"

        import cv2 as _cv2
        frame = _cv2.imread(str(img_path))
        if frame is None:
            unreadable += 1
            continue

        img_h, img_w = frame.shape[:2]

        # Run detector on GPU (it will be much faster, and we have enough VRAM)
        results = detector(
            frame,
            conf=CONF_THRESH,
            verbose=False,
            device=0,
        )
        boxes_raw = results[0].boxes
        dets = []
        if boxes_raw is not None and len(boxes_raw):
            for b in boxes_raw:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                dets.append((x1, y1, x2, y2, conf))

        target = _choose_box(dets, img_w, img_h)
        crop = None

        if target is not None:
            x1, y1, x2, y2, _ = target
            expanded = _expand_box(int(x1), int(y1), int(x2), int(y2), img_w, img_h)
            if expanded:
                ex1, ey1, ex2, ey2 = expanded
                crop = frame[ey1:ey2, ex1:ex2]
                detected += 1

        if crop is None or crop.size == 0:
            if not USE_FULL_WHEN_MISSING:
                missing += 1
                continue
            crop = frame
            missing += 1

        split = "val" if rng.random() < VAL_RATIO else "train"
        out_dir = crops_root / split / rel_class
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = img_path.stem
        out_path = out_dir / f"{stem}.jpg"
        idx = 1
        while out_path.exists():
            out_path = out_dir / f"{stem}_{idx}.jpg"
            idx += 1

        import cv2 as _cv2
        _cv2.imwrite(str(out_path), crop, [int(_cv2.IMWRITE_JPEG_QUALITY), 95])
        written += 1

    log(f"  Crops written     : {written}")
    log(f"  With detection    : {detected}")
    log(f"  Missing/fallback  : {missing}")
    log(f"  Unreadable        : {unreadable}")
    marker.touch()
    log("Phase A done.")


# ──────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────
class CropDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_samples(split_dir: Path, class_to_idx: dict):
    """Walk split_dir and return (path, int_label) pairs."""
    samples = []
    for img_path in _list_images(split_dir):
        rel = img_path.parent.relative_to(split_dir).as_posix()
        if rel == ".":
            rel = "unknown"
        if rel in class_to_idx:
            samples.append((img_path, class_to_idx[rel]))
    return samples


def make_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────
def train_stage(
    stage_name: str,
    train_samples,
    val_samples,
    class_names,
    save_path: Path,
    device,
    epochs: int,
):
    log("=" * 60)
    log(f"Training {stage_name}")
    log(f"  Classes    : {class_names}")
    log(f"  Train imgs : {len(train_samples)}  |  Val imgs: {len(val_samples)}")
    log(f"  Epochs     : {epochs}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")

    train_tf, val_tf = make_transforms()

    # Weighted sampler for class balance
    class_counts = defaultdict(int)
    for _, lbl in train_samples:
        class_counts[lbl] += 1
    sample_weights = [1.0 / max(1, class_counts[lbl]) for _, lbl in train_samples]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_ds = CropDataset(train_samples, train_tf)
    val_ds   = CropDataset(val_samples,   val_tf)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    # Model
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model = model.to(device)

    # Class weights for loss
    total = sum(class_counts.values())
    w = torch.tensor(
        [total / max(1, class_counts[i]) for i in range(len(class_names))],
        dtype=torch.float32,
    ).to(device)
    w = w / w.sum() * len(class_names)

    criterion  = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    optimizer  = optim.Adam(model.parameters(), lr=LR)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                      factor=0.5, patience=4)

    best_acc   = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        bar = tqdm(train_loader, desc=f"[{stage_name}] Ep {epoch}/{epochs}", leave=False)
        for imgs, lbls in bar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}", mem=gpu_mem_str())

        avg_loss = running_loss / max(1, len(train_loader))

        model.eval()
        correct = total_n = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                _, pred = torch.max(model(imgs), 1)
                total_n  += lbls.size(0)
                correct  += (pred == lbls).sum().item()

        acc = 100.0 * correct / max(1, total_n)
        scheduler.step(acc)

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        log(f"[{stage_name}] Ep {epoch:3d}/{epochs} | "
            f"Loss {avg_loss:.4f} | Val Acc {acc:.2f}% | "
            f"Best {best_acc:.2f}% | {gpu_mem_str()}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = model.state_dict()
    torch.save(best_state, save_path)
    log(f"Saved {stage_name} → {save_path}  (best val acc: {best_acc:.2f}%)")
    return best_acc


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    global _log_fh
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "a", encoding="utf-8")

    log("╔══════════════════════════════════════════════════════╗")
    log("║   Two-Stage Crop Classifier Training (YOLO + MN-V3)  ║")
    log("╚══════════════════════════════════════════════════════╝")

    # ── VRAM / swap setup ──────────────────────────────────────
    setup_vram_limit()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"GPU   : {p.name}  |  Total VRAM: {p.total_memory // (1024**2)} MB")

    # ── Phase A: crop dataset ──────────────────────────────────
    build_crops(CROPS_ROOT)

    # ── Discover classes in crop dataset ───────────────────────
    train_root = CROPS_ROOT / "train"
    val_root   = CROPS_ROOT / "val"

    # Collect all unique relative class dirs (e.g. "PMD", "Overige/Elektronica")
    all_classes_train = sorted({
        p.parent.relative_to(train_root).as_posix()
        for p in _list_images(train_root)
    })
    log(f"Classes found in crops/train: {all_classes_train}")

    # ── Stage 1: 5 main classes ─────────────────────────────────
    PREFERRED = ["Organisch", "PMD", "Papier", "Restafval", "Overige"]

    def to_main(c: str) -> str:
        return c.split("/")[0] if "/" in c else c

    main_classes_found = sorted({to_main(c) for c in all_classes_train})
    main_classes = [c for c in PREFERRED if c in main_classes_found]
    for c in main_classes_found:
        if c not in main_classes:
            main_classes.append(c)
    main_to_idx = {c: i for i, c in enumerate(main_classes)}

    # Build samples: map every crop-class to its main class index
    def make_main_samples(root: Path):
        samples = []
        for img_path in _list_images(root):
            rel = img_path.parent.relative_to(root).as_posix()
            main = to_main(rel) if rel != "." else "unknown"
            if main in main_to_idx:
                samples.append((img_path, main_to_idx[main]))
        return samples

    train_s1 = make_main_samples(train_root)
    val_s1   = make_main_samples(val_root)

    stage1_acc = train_stage(
        stage_name  = "Stage-1 (main classes)",
        train_samples = train_s1,
        val_samples   = val_s1,
        class_names   = main_classes,
        save_path     = OUTPUT_DIR / "stage1_main.pth",
        device        = device,
        epochs        = EPOCHS_STAGE1,
    )

    # ── Stage 2: Overige sub-classes ────────────────────────────
    overige_classes_found = sorted({
        c.split("/", 1)[1]
        for c in all_classes_train
        if c.startswith("Overige/")
    })

    if not overige_classes_found:
        log("No Overige/* sub-classes found — skipping Stage-2.")
        stage2_acc = None
    else:
        sub_to_idx = {c: i for i, c in enumerate(overige_classes_found)}

        def make_sub_samples(root: Path):
            samples = []
            for img_path in _list_images(root):
                rel = img_path.parent.relative_to(root).as_posix()
                if rel.startswith("Overige/"):
                    sub = rel.split("/", 1)[1]
                    if sub in sub_to_idx:
                        samples.append((img_path, sub_to_idx[sub]))
            return samples

        train_s2 = make_sub_samples(train_root)
        val_s2   = make_sub_samples(val_root)

        stage2_acc = train_stage(
            stage_name  = "Stage-2 (Overige sub-classes)",
            train_samples = train_s2,
            val_samples   = val_s2,
            class_names   = overige_classes_found,
            save_path     = OUTPUT_DIR / "stage2_overige.pth",
            device        = device,
            epochs        = EPOCHS_STAGE2,
        )

    # ── Save metadata ──────────────────────────────────────────
    metadata = {
        "input_size"            : IMG_SIZE,
        "stage1_classes"        : main_classes,
        "stage2_overige_classes": overige_classes_found if overige_classes_found else [],
        "stage1_model"          : "stage1_main.pth",
        "stage2_model"          : "stage2_overige.pth",
        "main_label_for_stage2" : "Overige",
        "default_fallback"      : "Restafval",
        "detector_used"         : str(BEST_DETECTOR),
        "crops_root"            : str(CROPS_ROOT),
        "vram_limit_mb"         : VRAM_LIMIT_MB,
        "val_accuracy": {
            "stage1_main"     : stage1_acc,
            "stage2_overige"  : stage2_acc,
        },
    }
    meta_path = OUTPUT_DIR / "two_stage_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log("=" * 60)
    log("Training complete.")
    log(f"  Stage-1 best val acc : {stage1_acc:.2f}%")
    if stage2_acc is not None:
        log(f"  Stage-2 best val acc : {stage2_acc:.2f}%")
    log(f"  Models saved to      : {OUTPUT_DIR}")
    log(f"  Metadata             : {meta_path}")
    log("To export ONNX: python export_two_stage_onnx.py")
    _log_fh.close()


if __name__ == "__main__":
    main()

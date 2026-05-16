import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_indexed_samples(root_dir: str):
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    samples = []
    classes_set = set()

    for dirpath, _, filenames in os.walk(root_dir):
        imgs = [f for f in filenames if f.lower().endswith(image_exts)]
        if not imgs:
            continue
        rel = os.path.relpath(dirpath, root_dir)
        classes_set.add(rel)
        for fname in imgs:
            samples.append((os.path.join(dirpath, fname), rel))

    classes = sorted(classes_set)
    return samples, classes


def stratified_split(labels, seed=42, train_ratio=0.8):
    by_class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        by_class_indices[label].append(idx)

    train_indices = []
    val_indices = []
    rng = random.Random(seed)

    for _, indices in by_class_indices.items():
        idxs = indices[:]
        rng.shuffle(idxs)
        split_at = int(train_ratio * len(idxs))
        if len(idxs) > 1:
            split_at = min(max(split_at, 1), len(idxs) - 1)
        train_indices.extend(idxs[:split_at])
        val_indices.extend(idxs[split_at:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


class PathLabelDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def train_single_stage(
    samples,
    class_names,
    save_path,
    device,
    epochs=25,
    batch_size=128,
    learning_rate=0.001,
    img_size=224,
    num_workers=8,
    random_seed=42,
):
    transform_train = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_val = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    labels = [label for _, label in samples]
    train_indices, val_indices = stratified_split(labels, seed=random_seed, train_ratio=0.8)

    dataset_train = PathLabelDataset(samples, transform=transform_train)
    dataset_val = PathLabelDataset(samples, transform=transform_val)
    train_dataset = Subset(dataset_train, train_indices)
    val_dataset = Subset(dataset_val, val_indices)

    class_counts = torch.zeros(len(class_names))
    for idx in train_indices:
        _, label = samples[idx]
        class_counts[label] += 1

    class_weights = 1.0 / class_counts.clamp(min=1.0)
    class_weights = class_weights / class_weights.sum() * len(class_names)
    class_weights = class_weights.to(device)

    sample_weights = [1.0 / class_counts[samples[idx][1]].item() for idx in train_indices]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model = model.to(device)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if torch.cuda.is_available() and gpu_count > 1:
        print(f"Multi-GPU actief via DataParallel op {gpu_count} GPU's")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_acc = -1.0
    best_state = None

    print(f"\nStart training ({os.path.basename(save_path)})")
    print(f"Klassen: {class_names}")
    print(f"Train samples: {len(train_indices)} | Val samples: {len(val_indices)}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=True)
        for images, labels_batch in train_bar:
            images, labels_batch = images.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()

        acc = 100.0 * correct / max(1, total)
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {k: v.cpu() for k, v in model_to_save.state_dict().items()}

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_train_loss:.4f} - Val Accuracy: {acc:.2f}%")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if best_state is None:
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        best_state = model_to_save.state_dict()
    torch.save(best_state, save_path)
    print(f"Best model opgeslagen: {save_path} (Val Accuracy: {best_acc:.2f}%)")
    return best_acc


def map_to_main_class(name: str) -> str:
    if name.startswith("Overige/"):
        return "Overige"
    return name


def main():
    data_root = "/root/smart_bin_project/data/Dataset/train"
    output_dir = "/root/smart_bin_project/models/two_stage"

    seed = 42
    epochs_stage1 = int(os.getenv("EPOCHS_STAGE1", "50")) # epoochs wijzigen
    epochs_stage2 = int(os.getenv("EPOCHS_STAGE2", "50"))
    batch_size = int(os.getenv("BATCH_SIZE", "128"))
    num_workers = int(os.getenv("NUM_WORKERS", "8"))
    learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))

    if not os.path.exists(data_root):
        print(f"FOUT: dataset niet gevonden op {data_root}")
        return

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"Aantal GPU's beschikbaar: {torch.cuda.device_count()}")

    raw_samples, raw_classes = build_indexed_samples(data_root)
    if not raw_samples:
        print("FOUT: geen afbeeldingen gevonden.")
        return

    # Stage 1: hoofdklassen (Organisch, PMD, Papier, Restafval, Overige)
    preferred_order = ["Organisch", "PMD", "Papier", "Restafval", "Overige"]
    main_classes_found = {map_to_main_class(c) for c in raw_classes}
    main_classes = [c for c in preferred_order if c in main_classes_found]
    for c in sorted(main_classes_found):
        if c not in main_classes:
            main_classes.append(c)
    main_to_idx = {c: i for i, c in enumerate(main_classes)}

    stage1_samples = []
    for path, label_name in raw_samples:
        main_name = map_to_main_class(label_name)
        stage1_samples.append((path, main_to_idx[main_name]))

    stage1_path = os.path.join(output_dir, "stage1_main.pth")
    stage1_acc = train_single_stage(
        samples=stage1_samples,
        class_names=main_classes,
        save_path=stage1_path,
        device=device,
        epochs=epochs_stage1,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_workers=num_workers,
        random_seed=seed,
    )

    # Stage 2: alleen subklassen binnen Overige/*
    sub_names = sorted({c.split("/", 1)[1] for c in raw_classes if c.startswith("Overige/")})
    if not sub_names:
        print("Geen Overige/* subklassen gevonden, stage 2 wordt overgeslagen.")
        return

    sub_to_idx = {c: i for i, c in enumerate(sub_names)}
    stage2_samples = []
    for path, label_name in raw_samples:
        if label_name.startswith("Overige/"):
            sub_name = label_name.split("/", 1)[1]
            stage2_samples.append((path, sub_to_idx[sub_name]))

    stage2_path = os.path.join(output_dir, "stage2_overige.pth")
    stage2_acc = train_single_stage(
        samples=stage2_samples,
        class_names=sub_names,
        save_path=stage2_path,
        device=device,
        epochs=epochs_stage2,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_workers=num_workers,
        random_seed=seed,
    )

    metadata = {
        "input_size": 224,
        "stage1_classes": main_classes,
        "stage2_overige_classes": sub_names,
        "stage1_model": "stage1_main.pth",
        "stage2_model": "stage2_overige.pth",
        "main_label_for_stage2": "Overige",
        "default_fallback": "Restafval",
        "val_accuracy": {
            "stage1_main": stage1_acc,
            "stage2_overige": stage2_acc,
        },
    }
    metadata_path = os.path.join(output_dir, "two_stage_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n2-staps training klaar.")
    print(f"Metadata opgeslagen: {metadata_path}")
    print("Gebruik voor inference: python pi_inference_two_stage.py <image_path>")


if __name__ == "__main__":
    main()

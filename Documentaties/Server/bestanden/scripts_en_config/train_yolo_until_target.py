#!/usr/bin/env python3
"""
Prepare and train a YOLO detector with automatic retries until a target metric is reached.

Key behavior:
- Builds a clean YOLO dataset layout from an image root + generated label root.
- Trains in attempts with progressively stronger settings.
- After each attempt, validates best weights and checks metric threshold.
- Stops when target metric is reached, or when max attempts is hit.

This script does not run by itself; execute it manually when you are ready.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

DEFAULT_CLASS_NAMES = ["Organisch", "PMD", "Papier", "Restafval", "Overige"]
DEFAULT_MODEL_CANDIDATES = [
    "/root/smart_bin_project/yolo26n.pt",
    "/root/smart_bin_project/yolov8n.pt",
    "yolov8s.pt",
]


@dataclass
class Sample:
    rel_path: str
    image_path: str
    label_path: str
    class_name: str
    box_count: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO with auto-retries until target metric is reached."
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="/root/smart_bin_project/data/Dataset/train",
        help="Root with source images.",
    )
    parser.add_argument(
        "--labels-root",
        type=str,
        default="/root/smart_bin_project/outputs/auto_labels_like_manual_parallel_clean",
        help="Root with YOLO .txt labels that match image relative paths.",
    )
    parser.add_argument(
        "--prepared-root",
        type=str,
        default="/root/smart_bin_project/outputs/yolo_prepared_from_generated_labels",
        help="Prepared YOLO dataset root (images/train|val and labels/train|val).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="/root/smart_bin_project/runs/detect_until_target",
        help="Ultralytics project output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="generated_labels_detector",
        help="Base run name; attempts get a suffix.",
    )
    parser.add_argument(
        "--history-json",
        type=str,
        default="",
        help="Optional history JSON path. Defaults to <project>/<run-name>_history.json",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma separated class names in class-id order.",
    )
    parser.add_argument(
        "--model-candidates",
        type=str,
        default=",".join(DEFAULT_MODEL_CANDIDATES),
        help="Comma separated model paths/names tried across attempts.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Train split ratio (rest is validation).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--clean-prepared",
        action="store_true",
        help="Delete prepared-root first, then rebuild.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking (slower, bigger, but portable).",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare dataset files, do not train.",
    )
    parser.add_argument(
        "--target-metric",
        type=float,
        default=0.96,
        help="Stop when selected metric >= this value.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="map50",
        choices=["map50", "map", "map75", "precision", "recall", "fitness"],
        help="Validation metric to optimize.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=8,
        help="Max training attempts. Use 0 for unlimited retries.",
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA id or cpu.")
    parser.add_argument("--workers", type=int, default=16, help="DataLoader workers.")
    parser.add_argument(
        "--cache",
        type=str,
        default="ram",
        choices=["none", "ram", "disk"],
        help="Ultralytics cache mode.",
    )
    parser.add_argument("--imgsz-base", type=int, default=640, help="Base image size.")
    parser.add_argument("--imgsz-max", type=int, default=1024, help="Max image size.")
    parser.add_argument(
        "--batch-base",
        type=int,
        default=64,
        help="Base batch size (reduced on harder attempts).",
    )
    parser.add_argument("--epochs-base", type=int, default=120, help="Base epochs.")
    parser.add_argument("--patience-base", type=int, default=40, help="Base patience.")
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=5.0,
        help="Pause between attempts (seconds).",
    )
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS])


def read_label_box_count(label_path: Path, num_classes: int) -> tuple[int, str | None]:
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as exc:
        return 0, f"read_error: {exc}"

    count = 0
    for idx, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            return 0, f"line_{idx}: expected 5 columns"
        try:
            cls = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            return 0, f"line_{idx}: non-numeric value"

        if cls < 0 or cls >= num_classes:
            return 0, f"line_{idx}: class_id {cls} out of range [0,{num_classes - 1}]"

        # Small tolerance for rounding noise.
        if not (-1e-3 <= x_center <= 1.001 and -1e-3 <= y_center <= 1.001):
            return 0, f"line_{idx}: center outside [0,1]"
        if not (1e-6 < width <= 1.001 and 1e-6 < height <= 1.001):
            return 0, f"line_{idx}: width/height outside (0,1]"

        count += 1

    if count == 0:
        return 0, "empty_or_no_valid_boxes"
    return count, None


def discover_samples(images_root: Path, labels_root: Path, class_names: list[str]) -> tuple[list[Sample], dict[str, Any]]:
    images = list_images(images_root)
    num_classes = len(class_names)

    stats: dict[str, Any] = {
        "images_scanned": len(images),
        "missing_label": 0,
        "invalid_label": 0,
        "usable_samples": 0,
        "boxes_total": 0,
        "class_counts": defaultdict(int),
        "invalid_examples": [],
    }

    samples: list[Sample] = []
    for i, img_path in enumerate(images, start=1):
        rel = img_path.relative_to(images_root)
        label_path = labels_root / rel.with_suffix(".txt")

        if not label_path.exists():
            stats["missing_label"] += 1
            continue

        box_count, err = read_label_box_count(label_path, num_classes=num_classes)
        if err is not None:
            stats["invalid_label"] += 1
            if len(stats["invalid_examples"]) < 20:
                stats["invalid_examples"].append(
                    {"label": str(label_path), "reason": err}
                )
            continue

        class_name = rel.parts[0] if rel.parts else "__root__"
        sample = Sample(
            rel_path=rel.as_posix(),
            image_path=str(img_path),
            label_path=str(label_path),
            class_name=class_name,
            box_count=box_count,
        )
        samples.append(sample)
        stats["usable_samples"] += 1
        stats["boxes_total"] += box_count
        stats["class_counts"][class_name] += 1

        if i % 3000 == 0:
            print(
                f"[scan] {i}/{len(images)} images checked | usable={stats['usable_samples']} "
                f"missing={stats['missing_label']} invalid={stats['invalid_label']}"
            )

    stats["class_counts"] = dict(sorted(stats["class_counts"].items(), key=lambda kv: kv[0]))
    return samples, stats


def stratified_split(samples: list[Sample], train_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")

    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.class_name].append(sample)

    rng = random.Random(seed)
    train_split: list[Sample] = []
    val_split: list[Sample] = []

    for class_name, class_items in grouped.items():
        rng.shuffle(class_items)
        n = len(class_items)
        n_train = int(round(n * train_ratio))
        if n >= 2:
            n_train = max(1, min(n - 1, n_train))
        else:
            n_train = 1

        train_split.extend(class_items[:n_train])
        val_split.extend(class_items[n_train:])
        print(
            f"[split] {class_name}: total={n}, train={len(class_items[:n_train])}, val={len(class_items[n_train:])}"
        )

    rng.shuffle(train_split)
    rng.shuffle(val_split)
    return train_split, val_split


def copy_or_link(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy_files:
        shutil.copy2(src, dst)
        return

    try:
        os.symlink(src, dst)
    except OSError:
        # Filesystem may not support symlinks; fallback to copy.
        shutil.copy2(src, dst)


def prepare_dataset(
    prepared_root: Path,
    train_samples: list[Sample],
    val_samples: list[Sample],
    class_names: list[str],
    copy_images: bool,
    clean_prepared: bool,
) -> Path:
    if clean_prepared and prepared_root.exists():
        print(f"[prepare] cleaning {prepared_root}")
        shutil.rmtree(prepared_root)

    for split_name, split_samples in (("train", train_samples), ("val", val_samples)):
        for idx, sample in enumerate(split_samples, start=1):
            rel = Path(sample.rel_path)
            src_img = Path(sample.image_path)
            src_lbl = Path(sample.label_path)

            dst_img = prepared_root / "images" / split_name / rel
            dst_lbl = prepared_root / "labels" / split_name / rel.with_suffix(".txt")

            copy_or_link(src_img, dst_img, copy_files=copy_images)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_lbl, dst_lbl)

            if idx % 4000 == 0:
                print(f"[prepare] {split_name}: {idx}/{len(split_samples)}")

    yaml_path = prepared_root / "dataset.yaml"
    payload = {
        "path": str(prepared_root),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    train_list = prepared_root / "train_images.txt"
    val_list = prepared_root / "val_images.txt"
    train_list.write_text(
        "\n".join(str((prepared_root / "images" / "train" / Path(s.rel_path)).resolve()) for s in train_samples)
        + "\n",
        encoding="utf-8",
    )
    val_list.write_text(
        "\n".join(str((prepared_root / "images" / "val" / Path(s.rel_path)).resolve()) for s in val_samples) + "\n",
        encoding="utf-8",
    )

    return yaml_path


def extract_metric(metrics: Any, metric: str) -> float:
    results_dict = getattr(metrics, "results_dict", None)
    if not isinstance(results_dict, dict):
        results_dict = {}

    def from_results_dict(keys: list[str]) -> float | None:
        for key in keys:
            val = results_dict.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    box_metrics = getattr(metrics, "box", None)

    if metric == "map50":
        if box_metrics is not None and hasattr(box_metrics, "map50"):
            return float(box_metrics.map50)
        val = from_results_dict(["metrics/mAP50(B)", "metrics/mAP50"])
        if val is not None:
            return val
    elif metric == "map":
        if box_metrics is not None and hasattr(box_metrics, "map"):
            return float(box_metrics.map)
        val = from_results_dict(["metrics/mAP50-95(B)", "metrics/mAP50-95"])
        if val is not None:
            return val
    elif metric == "map75":
        if box_metrics is not None and hasattr(box_metrics, "map75"):
            return float(box_metrics.map75)
        val = from_results_dict(["metrics/mAP75(B)", "metrics/mAP75"])
        if val is not None:
            return val
    elif metric == "precision":
        if box_metrics is not None and hasattr(box_metrics, "mp"):
            return float(box_metrics.mp)
        val = from_results_dict(["metrics/precision(B)", "metrics/precision"])
        if val is not None:
            return val
    elif metric == "recall":
        if box_metrics is not None and hasattr(box_metrics, "mr"):
            return float(box_metrics.mr)
        val = from_results_dict(["metrics/recall(B)", "metrics/recall"])
        if val is not None:
            return val
    elif metric == "fitness":
        if hasattr(metrics, "fitness"):
            fitness = getattr(metrics, "fitness")
            try:
                return float(fitness() if callable(fitness) else fitness)
            except (TypeError, ValueError):
                pass
        val = from_results_dict(["fitness"])
        if val is not None:
            return val

    raise RuntimeError(
        f"Could not extract metric '{metric}' from validation output. "
        f"Available keys: {sorted(results_dict.keys())}"
    )


def resolve_best_weights(train_result: Any, project: Path, run_name: str) -> Path | None:
    candidates: list[Path] = []

    save_dir = getattr(train_result, "save_dir", None)
    if save_dir:
        candidates.append(Path(save_dir) / "weights" / "best.pt")
        candidates.append(Path(save_dir) / "weights" / "last.pt")

    run_dir = project / run_name
    candidates.append(run_dir / "weights" / "best.pt")
    candidates.append(run_dir / "weights" / "last.pt")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def cache_arg(raw: str) -> bool | str:
    if raw == "none":
        return False
    return raw


def attempt_config(
    attempt_number: int,
    model_candidates: list[str],
    imgsz_base: int,
    imgsz_max: int,
    batch_base: int,
    epochs_base: int,
    patience_base: int,
) -> dict[str, Any]:
    model_idx = (attempt_number - 1) % len(model_candidates)
    cycle = (attempt_number - 1) // len(model_candidates)

    imgsz = min(imgsz_max, imgsz_base + 64 * cycle)
    epochs = epochs_base + 35 * cycle
    patience = patience_base + 10 * cycle
    batch_divisor = 1 + cycle // 2
    batch = max(4, int(batch_base // batch_divisor))

    # Train policy gets slightly stronger after each cycle.
    lr0 = 0.01 if cycle == 0 else (0.005 if cycle == 1 else 0.003)
    lrf = 0.01 if cycle == 0 else 0.001
    mixup = min(0.25, 0.10 + 0.05 * cycle)
    copy_paste = min(0.20, 0.05 + 0.05 * cycle)
    close_mosaic = max(5, 10 - cycle)
    optimizer = "SGD" if cycle < 2 else "AdamW"

    return {
        "tag": f"a{attempt_number:02d}_c{cycle}_m{model_idx}",
        "model": model_candidates[model_idx],
        "imgsz": imgsz,
        "batch": batch,
        "epochs": epochs,
        "patience": patience,
        "lr0": lr0,
        "lrf": lrf,
        "mixup": mixup,
        "copy_paste": copy_paste,
        "close_mosaic": close_mosaic,
        "optimizer": optimizer,
        "cycle": cycle,
    }


def write_history(history_path: Path, payload: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    images_root = Path(args.images_root).resolve()
    labels_root = Path(args.labels_root).resolve()
    prepared_root = Path(args.prepared_root).resolve()
    project_root = Path(args.project).resolve()

    class_names = parse_csv_list(args.class_names)
    model_candidates = parse_csv_list(args.model_candidates)
    if not class_names:
        raise ValueError("No class names provided.")
    if not model_candidates:
        raise ValueError("No model candidates provided.")

    if not images_root.exists():
        raise FileNotFoundError(f"images root not found: {images_root}")
    if not labels_root.exists():
        raise FileNotFoundError(f"labels root not found: {labels_root}")

    history_path = (
        Path(args.history_json).resolve()
        if args.history_json.strip()
        else (project_root / f"{args.run_name}_history.json")
    )

    existing_models = [m for m in model_candidates if Path(m).exists()]
    print("Configuration")
    print(f"- images_root      : {images_root}")
    print(f"- labels_root      : {labels_root}")
    print(f"- prepared_root    : {prepared_root}")
    print(f"- project_root     : {project_root}")
    print(f"- class_names      : {class_names}")
    print(f"- model_candidates : {model_candidates}")
    print(f"- existing_models  : {existing_models if existing_models else 'none_local'}")
    print(f"- target_metric    : {args.metric} >= {args.target_metric}")
    print(f"- max_attempts     : {args.max_attempts} (0 means unlimited)")
    print(f"- device           : {args.device}")
    print("")

    print("Scanning images + labels...")
    samples, scan_stats = discover_samples(images_root, labels_root, class_names=class_names)
    if not samples:
        raise RuntimeError("No usable labeled samples found. Check --images-root and --labels-root.")

    print("Scan summary")
    print(f"- images_scanned   : {scan_stats['images_scanned']}")
    print(f"- usable_samples   : {scan_stats['usable_samples']}")
    print(f"- missing_label    : {scan_stats['missing_label']}")
    print(f"- invalid_label    : {scan_stats['invalid_label']}")
    print(f"- boxes_total      : {scan_stats['boxes_total']}")
    print(f"- per_class        : {scan_stats['class_counts']}")
    if scan_stats["invalid_examples"]:
        print("- invalid_examples :")
        for row in scan_stats["invalid_examples"][:10]:
            print(f"  {row['label']} -> {row['reason']}")
    print("")

    print("Creating stratified train/val split...")
    train_split, val_split = stratified_split(samples, train_ratio=args.train_ratio, seed=args.seed)
    if not train_split or not val_split:
        raise RuntimeError(
            f"Invalid split: train={len(train_split)}, val={len(val_split)}. "
            f"Adjust --train-ratio or inspect class balance."
        )
    print(f"Split summary: train={len(train_split)} | val={len(val_split)}")
    print("")

    print("Preparing YOLO dataset layout...")
    dataset_yaml = prepare_dataset(
        prepared_root=prepared_root,
        train_samples=train_split,
        val_samples=val_split,
        class_names=class_names,
        copy_images=args.copy_images,
        clean_prepared=args.clean_prepared,
    )
    print(f"Prepared dataset YAML: {dataset_yaml}")
    print("")

    base_summary: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "images_root": str(images_root),
        "labels_root": str(labels_root),
        "prepared_root": str(prepared_root),
        "dataset_yaml": str(dataset_yaml),
        "project_root": str(project_root),
        "run_name": args.run_name,
        "class_names": class_names,
        "model_candidates": model_candidates,
        "target_metric_name": args.metric,
        "target_metric_value": args.target_metric,
        "max_attempts": args.max_attempts,
        "device": args.device,
        "split": {"train": len(train_split), "val": len(val_split)},
        "scan_stats": scan_stats,
        "attempts": [],
        "status": "prepared_only" if args.prepare_only else "running",
        "best_attempt": None,
    }
    write_history(history_path, base_summary)

    if args.prepare_only:
        print("Prepare-only mode complete. Training was not started.")
        print(f"History JSON: {history_path}")
        return

    reached_target = False
    best_score = -1.0
    best_attempt_record: dict[str, Any] | None = None

    attempt_number = 1
    while args.max_attempts == 0 or attempt_number <= args.max_attempts:
        config = attempt_config(
            attempt_number=attempt_number,
            model_candidates=model_candidates,
            imgsz_base=args.imgsz_base,
            imgsz_max=args.imgsz_max,
            batch_base=args.batch_base,
            epochs_base=args.epochs_base,
            patience_base=args.patience_base,
        )
        attempt_run_name = f"{args.run_name}_{config['tag']}"
        attempt_seed = args.seed + (attempt_number * 31)
        attempt_start = time.time()

        print("=" * 80)
        print(f"Attempt {attempt_number}")
        print(f"- run_name       : {attempt_run_name}")
        print(f"- model          : {config['model']}")
        print(f"- imgsz          : {config['imgsz']}")
        print(f"- batch          : {config['batch']}")
        print(f"- epochs         : {config['epochs']}")
        print(f"- patience       : {config['patience']}")
        print(f"- optimizer      : {config['optimizer']}")
        print(f"- seed           : {attempt_seed}")
        print("=" * 80)

        record: dict[str, Any] = {
            "attempt": attempt_number,
            "config": config,
            "run_name": attempt_run_name,
            "seed": attempt_seed,
            "started_at": utc_now_iso(),
            "status": "failed",
            "error": None,
            "best_weights": None,
            "metric_name": args.metric,
            "metric_value": None,
            "duration_seconds": None,
            "target_reached": False,
        }

        try:
            trainer = YOLO(config["model"])
            train_result = trainer.train(
                data=str(dataset_yaml),
                epochs=config["epochs"],
                imgsz=config["imgsz"],
                batch=config["batch"],
                project=str(project_root),
                name=attempt_run_name,
                device=args.device,
                workers=args.workers,
                cache=cache_arg(args.cache),
                patience=config["patience"],
                seed=attempt_seed,
                optimizer=config["optimizer"],
                lr0=config["lr0"],
                lrf=config["lrf"],
                weight_decay=0.0005,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                cos_lr=config["cycle"] >= 1,
                close_mosaic=config["close_mosaic"],
                hsv_h=0.015,
                hsv_s=0.70,
                hsv_v=0.40,
                degrees=5.0,
                translate=0.10,
                scale=0.50,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.50,
                mosaic=1.0,
                mixup=config["mixup"],
                copy_paste=config["copy_paste"],
                deterministic=False,
                amp=True,
                plots=True,
                verbose=True,
            )

            best_weights = resolve_best_weights(
                train_result=train_result,
                project=project_root,
                run_name=attempt_run_name,
            )
            if best_weights is None:
                raise RuntimeError("Could not locate best/last weights after training.")

            evaluator = YOLO(str(best_weights))
            val_metrics = evaluator.val(
                data=str(dataset_yaml),
                split="val",
                imgsz=config["imgsz"],
                batch=max(1, config["batch"] // 2),
                device=args.device,
                workers=max(1, args.workers // 2),
                verbose=False,
                plots=False,
            )
            score = extract_metric(val_metrics, metric=args.metric)

            record["status"] = "ok"
            record["best_weights"] = str(best_weights)
            record["metric_value"] = score
            record["target_reached"] = score >= args.target_metric
            print(
                f"Attempt {attempt_number} metric: {args.metric}={score:.6f} "
                f"(target {args.target_metric:.6f})"
            )

            if score > best_score:
                best_score = score
                best_attempt_record = record

            if score >= args.target_metric:
                reached_target = True

        except Exception as exc:
            record["error"] = str(exc)
            print(f"Attempt {attempt_number} failed: {exc}")
        finally:
            record["duration_seconds"] = round(time.time() - attempt_start, 2)
            record["finished_at"] = utc_now_iso()

            base_summary["attempts"].append(record)
            base_summary["best_attempt"] = best_attempt_record
            base_summary["status"] = "target_reached" if reached_target else "running"
            write_history(history_path, base_summary)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if reached_target:
            break

        attempt_number += 1
        if args.cooldown_seconds > 0:
            print(f"Cooling down for {args.cooldown_seconds:.1f}s before next attempt...")
            time.sleep(args.cooldown_seconds)

    base_summary["finished_at"] = utc_now_iso()
    base_summary["status"] = "target_reached" if reached_target else "max_attempts_reached"
    base_summary["best_attempt"] = best_attempt_record
    write_history(history_path, base_summary)

    print("")
    print("Training loop finished.")
    print(f"- status       : {base_summary['status']}")
    print(f"- history_json : {history_path}")
    if best_attempt_record is not None:
        print(
            f"- best_metric  : {best_attempt_record['metric_name']}="
            f"{best_attempt_record['metric_value']}"
        )
        print(f"- best_weights : {best_attempt_record['best_weights']}")
    else:
        print("- best_metric  : not available")


if __name__ == "__main__":
    main()


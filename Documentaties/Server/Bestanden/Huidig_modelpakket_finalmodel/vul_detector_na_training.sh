#!/bin/bash
set -euo pipefail

RUN_DIR="${1:-/root/smart_bin_project/runs/yolo_light/garbage_detector_pi_light}"
OUT_DIR="/root/smart_bin_project/test10mei_light"
BEST_PT="$RUN_DIR/weights/best.pt"
BEST_ONNX="$RUN_DIR/weights/best.onnx"

if [[ ! -f "$BEST_PT" ]]; then
  echo "FOUT: best.pt niet gevonden op $BEST_PT"
  echo "Wacht tot de training klaar is of geef een andere runmap mee."
  exit 1
fi

echo "==> Kopieer best.pt"
cp "$BEST_PT" "$OUT_DIR/best.pt"

echo "==> Exporteer detector naar ONNX"
yolo export model="$BEST_PT" format=onnx imgsz=640

if [[ ! -f "$BEST_ONNX" ]]; then
  echo "FOUT: ONNX export lijkt niet gelukt. Verwacht bestand: $BEST_ONNX"
  exit 1
fi

echo "==> Kopieer ONNX detector"
cp "$BEST_ONNX" "$OUT_DIR/yolov8_detector.onnx"

echo "Klaar."
echo "Bestanden in $OUT_DIR:"
ls -lh "$OUT_DIR"

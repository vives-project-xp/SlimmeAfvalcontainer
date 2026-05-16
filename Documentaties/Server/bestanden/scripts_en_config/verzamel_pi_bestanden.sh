#!/bin/bash
set -e

echo "=== 1. ONNX Modellen genereren ==="
python3 export_two_stage_onnx.py
echo ""

echo "=== 2. Bestanden verzamelen ==="
DEPLOY_DIR="/root/smart_bin_project/pi_deploy_pakket"

# Kopieer ONNX modellen en metadata
cp /root/smart_bin_project/models/two_stage_crops/*.onnx $DEPLOY_DIR/
cp /root/smart_bin_project/models/two_stage_crops/*.onnx.data $DEPLOY_DIR/ 2>/dev/null || true
cp /root/smart_bin_project/models/two_stage_crops/two_stage_metadata.json $DEPLOY_DIR/

# Exporteer YOLO model (best.pt) naar ONNX en kopieer deze
yolo export model=/root/smart_bin_project/runs/detect_strong/garbage_detector_l_fallback_aware_768-6/weights/best.pt format=onnx imgsz=768
cp /root/smart_bin_project/runs/detect_strong/garbage_detector_l_fallback_aware_768-6/weights/best.onnx $DEPLOY_DIR/yolov8_detector.onnx

# Kopieer Inference Scripts
cp /root/smart_bin_project/pi_inference_two_stage.py $DEPLOY_DIR/
cp /root/smart_bin_project/main.py $DEPLOY_DIR/
cp /root/smart_bin_project/classifier.py $DEPLOY_DIR/
cp /root/smart_bin_project/detector.py $DEPLOY_DIR/

echo "Klaar! Alle bestanden voor de Pi staan nu in: $DEPLOY_DIR"
ls -lh $DEPLOY_DIR

# 6. Exporteren en modelpakket maken

Na training moeten de modellen naar ONNX en moeten alle nodige bestanden samen in een pakketmap komen.

Korte uitleg:

- `.pth`: PyTorch classifiermodel na training.
- `.pt`: YOLO/PyTorch detectorcheckpoint.
- `.onnx`: exportformaat om modellen makkelijker buiten PyTorch te gebruiken.
- `.onnx.data`: extra databestand dat bij sommige ONNX exports hoort.

Opmerking: de detectorexport heet in het huidige pakket `yolov8_detector.onnx`. Dat is een bestaande bestandsnaam. De handleiding bedoelt hiermee de YOLO-detector export, niet noodzakelijk een algemene keuze om alleen over YOLOv8 te spreken.

## 6.1 Classifier exporteren

Script:

[export_two_stage_onnx.py](../../Bestanden/Scripts_en_config/export_two_stage_onnx.py)

Run:

```bash
cd /root/smart_bin_project
source .venv/bin/activate
python3 export_two_stage_onnx.py
```

Input:

```text
/root/smart_bin_project/models/two_stage_crops/stage1_main.pth
/root/smart_bin_project/models/two_stage_crops/stage2_overige.pth
/root/smart_bin_project/models/two_stage_crops/two_stage_metadata.json
```

Referentie van de outputs:

- [Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)
- [two_stage_training_outputs/README.md](../../Bestanden/two_stage_training_outputs/README.md)

Let op: de `.onnx.data` bestanden zijn noodzakelijk.

## 6.2 Detector exporteren

```bash
yolo export model=/root/smart_bin_project/runs/detect_strong/garbage_detector_l_fallback_aware_768-6/weights/best.pt format=onnx imgsz=768
```

Kopieer of hernoem daarna:

```text
best.onnx -> yolov8_detector.onnx
```

Gebruik die naam omdat de bestaande scripts dit bestand zo verwachten.

Referentie:

[yolov8_detector.onnx](../../Bestanden/Huidig_modelpakket_finalmodel/yolov8_detector.onnx)

## 6.3 Modelpakketmap maken

Maak een pakketmap:

```bash
mkdir -p /root/smart_bin_project/test_reproductie_light_crops
```

Kopieer de classifier exports:

```bash
cp /root/smart_bin_project/models/two_stage_crops/stage1_main.onnx* /root/smart_bin_project/test_reproductie_light_crops/
cp /root/smart_bin_project/models/two_stage_crops/stage2_overige.onnx* /root/smart_bin_project/test_reproductie_light_crops/
cp /root/smart_bin_project/models/two_stage_crops/two_stage_metadata.json /root/smart_bin_project/test_reproductie_light_crops/
```

Kopieer detector:

```bash
cp /root/smart_bin_project/runs/detect_strong/garbage_detector_l_fallback_aware_768-6/weights/best.pt /root/smart_bin_project/test_reproductie_light_crops/
cp /root/smart_bin_project/runs/detect_strong/garbage_detector_l_fallback_aware_768-6/weights/best.onnx /root/smart_bin_project/test_reproductie_light_crops/yolov8_detector.onnx
```

## 6.4 Compleet pakket

Minimaal:

- [best.pt](../../Bestanden/Huidig_modelpakket_finalmodel/best.pt)
- [yolov8_detector.onnx](../../Bestanden/Huidig_modelpakket_finalmodel/yolov8_detector.onnx)
- [stage1_main.onnx](../../Bestanden/Huidig_modelpakket_finalmodel/stage1_main.onnx)
- [stage1_main.onnx.data](../../Bestanden/Huidig_modelpakket_finalmodel/stage1_main.onnx.data)
- [stage2_overige.onnx](../../Bestanden/Huidig_modelpakket_finalmodel/stage2_overige.onnx)
- [stage2_overige.onnx.data](../../Bestanden/Huidig_modelpakket_finalmodel/stage2_overige.onnx.data)
- [two_stage_metadata.json](../../Bestanden/Huidig_modelpakket_finalmodel/two_stage_metadata.json)
- [classifier.py](../../Bestanden/Huidig_modelpakket_finalmodel/classifier.py)
- [detector.py](../../Bestanden/Huidig_modelpakket_finalmodel/detector.py)
- [main.py](../../Bestanden/Huidig_modelpakket_finalmodel/main.py)
- [pi_inference_two_stage.py](../../Bestanden/Huidig_modelpakket_finalmodel/pi_inference_two_stage.py)

Voor de volledige trainingsoutputs gebruik je daarnaast:

- [Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)

## 6.5 Verzamelscript

Er is ook een bestaand verzamelscript:

[verzamel_pi_bestanden.sh](../../Bestanden/Scripts_en_config/verzamel_pi_bestanden.sh)


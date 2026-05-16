# 6. Exporteren en modelpakket maken

Na training moeten de modellen naar ONNX en moeten alle nodige bestanden samen
in een pakketmap komen.

Korte uitleg:

- `.pth`: PyTorch classifiermodel na training.
- `.pt`: YOLO/PyTorch detectorcheckpoint.
- `.onnx`: exportformaat om modellen makkelijker buiten PyTorch te gebruiken.
- `.onnx.data`: extra databestand dat bij sommige ONNX exports hoort.

Opmerking: de detectorexport heet in het huidige pakket
`yolov8_detector.onnx`. Dat is een bestaande/legacy bestandsnaam. De handleiding
bedoelt hiermee de YOLO-detector export, niet noodzakelijk een algemene keuze om
alleen over YOLOv8 te spreken.

## 6.1 Classifier exporteren

Script:

[export_two_stage_onnx.py](../../bestanden/scripts_en_config/export_two_stage_onnx.py)

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

Output:

- [stage1_main.onnx](../../bestanden/two_stage_training_outputs/stage1_main.onnx)
- [stage1_main.onnx.data](../../bestanden/two_stage_training_outputs/stage1_main.onnx.data)
- [stage2_overige.onnx](../../bestanden/two_stage_training_outputs/stage2_overige.onnx)
- [stage2_overige.onnx.data](../../bestanden/two_stage_training_outputs/stage2_overige.onnx.data)

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

[yolov8_detector.onnx](../../bestanden/huidig_modelpakket_test11mei_light_crops/yolov8_detector.onnx)

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

Kopieer inference/helpers:

```bash
cp /root/smart_bin_project/classifier.py /root/smart_bin_project/test_reproductie_light_crops/
cp /root/smart_bin_project/detector.py /root/smart_bin_project/test_reproductie_light_crops/
cp /root/smart_bin_project/main.py /root/smart_bin_project/test_reproductie_light_crops/
cp /root/smart_bin_project/pi_inference_two_stage.py /root/smart_bin_project/test_reproductie_light_crops/
```

## 6.4 Compleet pakket

Minimaal:

- [best.pt](../../bestanden/huidig_modelpakket_test11mei_light_crops/best.pt)
- [yolov8_detector.onnx](../../bestanden/huidig_modelpakket_test11mei_light_crops/yolov8_detector.onnx)
- [stage1_main.onnx](../../bestanden/huidig_modelpakket_test11mei_light_crops/stage1_main.onnx)
- [stage1_main.onnx.data](../../bestanden/huidig_modelpakket_test11mei_light_crops/stage1_main.onnx.data)
- [stage2_overige.onnx](../../bestanden/huidig_modelpakket_test11mei_light_crops/stage2_overige.onnx)
- [stage2_overige.onnx.data](../../bestanden/huidig_modelpakket_test11mei_light_crops/stage2_overige.onnx.data)
- [two_stage_metadata.json](../../bestanden/huidig_modelpakket_test11mei_light_crops/two_stage_metadata.json)
- [classifier.py](../../bestanden/huidig_modelpakket_test11mei_light_crops/classifier.py)
- [detector.py](../../bestanden/huidig_modelpakket_test11mei_light_crops/detector.py)
- [main.py](../../bestanden/huidig_modelpakket_test11mei_light_crops/main.py)
- [pi_inference_two_stage.py](../../bestanden/huidig_modelpakket_test11mei_light_crops/pi_inference_two_stage.py)

Optioneel of voor latere Pi/Hailo flow:

- [hailo_two_stage_main.py](../../bestanden/huidig_modelpakket_test11mei_light_crops/hailo_two_stage_main.py)
- [vul_detector_na_training.sh](../../bestanden/huidig_modelpakket_test11mei_light_crops/vul_detector_na_training.sh)

## 6.5 Verzamelscript

Er is ook een bestaand verzamelscript:

[verzamel_pi_bestanden.sh](../../bestanden/scripts_en_config/verzamel_pi_bestanden.sh)

Gebruik dit alleen als de paden in het script overeenkomen met jouw project.

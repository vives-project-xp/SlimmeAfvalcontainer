# Smart Bin modeltraining documentatie

Deze map bevat de documentatie en bestanden om het huidige smart-bin trainingsresultaat serverkant te begrijpen en opnieuw te maken.

Het gaat hier over de training en export van de modellen, niet over de installatie op de Raspberry Pi zelf.

## Inhoud

```text
documentatie/
|-- README.md
|-- Handleiding/
|   |-- README.md
|   |-- 01_vm_en_project/
|   |-- 02_dependencies/
|   |-- 03_dataset/
|   |-- 04_detector/
|   |-- 05_crop_classifier/
|   |-- 06_export_modelpakket/
|   `-- 07_controle_en_github/
`-- Bestanden/
    |-- README.md
    |-- MANIFEST.txt
    |-- LOCAL_MANIFEST.txt
    |-- Huidig_modelpakket_test11mei_light_crops/
    |-- two_stage_training_outputs/
    |-- Gebruikte_detector/
    `-- Scripts_en_config/
```

## Waarvoor dient deze map?

- [Handleiding/](Handleiding/README.md) legt stap voor stap uit hoe je de dataset, detector, cropdataset, two-stage classifier en exports opnieuw maakt.
- [Bestanden/](Bestanden/README.md) bevat de echte referentiebestanden, scripts, configs en metadata die bij het huidige resultaat horen.

## Huidig referentiemodel

Het modelpakket dat nu als referentie gebruikt wordt:

```text
/root/smart_bin_project/test11mei_light_crops
```

In deze documentatiemap staat dat onder:

[Bestanden/Huidig_modelpakket_test11mei_light_crops](Bestanden/Huidig_modelpakket_test11mei_light_crops/)

## Snel starten

1. Lees [Handleiding/README.md](Handleiding/README.md).
2. Begin vanaf nul met [01_vm_en_project](Handleiding/01_vm_en_project/README.md).
3. Controleer [Bestanden/README.md](Bestanden/README.md).
4. Gebruik [Bestanden/MANIFEST.txt](Bestanden/MANIFEST.txt) om te zien welke artifacts bij het referentieresultaat horen.
5. Controleer [Bestanden/LOCAL_MANIFEST.txt](Bestanden/LOCAL_MANIFEST.txt) voor de GitHub-versie van deze map.

## Handleidingen per onderdeel

- [VM en projectmap maken](Handleiding/01_vm_en_project/README.md)
- [Dependencies installeren](Handleiding/02_dependencies/README.md)
- [Dataset voorbereiden](Handleiding/03_dataset/README.md)
- [YOLO-detector trainen of klaarzetten](Handleiding/04_detector/README.md)
- [Cropdataset en two-stage classifier trainen](Handleiding/05_crop_classifier/README.md)
- [Exporteren en modelpakket maken](Handleiding/06_export_modelpakket/README.md)
- [Controle, GitHub en grote bestanden](Handleiding/07_controle_en_github/README.md)

## Snelle links naar belangrijke bestanden en downloads

- Finale cropdataset op Kaggle: [Smart Bin Classifier Crops](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-classifier-crops)
- Originele foto's op Kaggle: [Smart Bin Original Images](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-original-images)
- Zware model-artifacts op Kaggle: [Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)
- Huidige metadata: [two_stage_metadata.json](Bestanden/Huidig_modelpakket_test11mei_light_crops/two_stage_metadata.json)
- Huidige detector in het pakket: [best.pt](Bestanden/Huidig_modelpakket_test11mei_light_crops/best.pt)
- Detector gebruikt om crops te bouwen: [Gebruikte_detector/README.md](Bestanden/Gebruikte_detector/README.md)
- Two-stage trainingsoutputs: [two_stage_training_outputs/README.md](Bestanden/two_stage_training_outputs/README.md)
- Detector ONNX: [yolov8_detector.onnx](Bestanden/Huidig_modelpakket_test11mei_light_crops/yolov8_detector.onnx)
- Stage 1 ONNX: [stage1_main.onnx](Bestanden/Huidig_modelpakket_test11mei_light_crops/stage1_main.onnx)
- Stage 2 ONNX: [stage2_overige.onnx](Bestanden/Huidig_modelpakket_test11mei_light_crops/stage2_overige.onnx)
- Trainingsscript crops: [train_two_stage_crops.py](Bestanden/Scripts_en_config/train_two_stage_crops.py)
- Exportscript ONNX: [export_two_stage_onnx.py](Bestanden/Scripts_en_config/export_two_stage_onnx.py)
- Requirements: [requirements.txt](Bestanden/Scripts_en_config/requirements.txt)

Opmerking: `yolov8_detector.onnx` is de bestandsnaam die in het huidige pakket gebruikt wordt. In de uitleg noemen we dit gewoon de YOLO-detector. Er staan ook YOLOv11-runs op de VM, maar de metadata van het huidige finale pakket verwijst naar de detectorrun `garbage_detector_l_fallback_aware_768-6`.

## Belangrijke GitHub-opmerking

Deze repo bevat bewust niet alle zware trainingsbestanden als gewone Git-bestanden.

De gekozen verdeling is:

- GitHub voor code, documentatie, scripts, configs en het compacte referentiepakket
- Kaggle voor originele foto's, crops en zware model-artifacts

Daardoor blijven de links werkbaar en kan het project toch volledig opnieuw opgebouwd worden.

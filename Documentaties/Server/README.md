# Smart Bin modeltraining documentatie

Deze map bevat de documentatie en bestanden om het huidige smart-bin
trainingsresultaat serverkant te begrijpen en opnieuw te maken.

Het gaat hier over de training en export van de modellen, niet over de installatie
op de Raspberry Pi zelf.

## Inhoud

```text
documentatie/
|-- README.md
|-- handleiding/
|   |-- README.md
|   |-- 01_vm_en_project/
|   |-- 02_dependencies/
|   |-- 03_dataset/
|   |-- 04_detector/
|   |-- 05_crop_classifier/
|   |-- 06_export_modelpakket/
|   `-- 07_controle_en_github/
`-- bestanden/
    |-- README.md
    |-- MANIFEST.txt
    |-- huidig_modelpakket_test11mei_light_crops/
    |-- two_stage_training_outputs/
    |-- gebruikte_detector/
    `-- scripts_en_config/
```

## Waarvoor dient deze map?

- [handleiding/](handleiding/README.md) legt stap voor stap uit hoe je de
  dataset, detector, cropdataset, two-stage classifier en exports opnieuw maakt.
- [bestanden/](bestanden/README.md) bevat de echte modelbestanden, scripts,
  configs en metadata die bij het huidige resultaat horen.

## Huidig referentiemodel

Het modelpakket dat nu als referentie gebruikt wordt:

```text
/root/smart_bin_project/test11mei_light_crops
```

In deze documentatiemap staat dat onder:

[bestanden/huidig_modelpakket_test11mei_light_crops](bestanden/huidig_modelpakket_test11mei_light_crops/)

## Snel starten

1. Lees [handleiding/README.md](handleiding/README.md).
2. Begin vanaf nul met [01_vm_en_project](handleiding/01_vm_en_project/README.md).
3. Controleer [bestanden/README.md](bestanden/README.md).
4. Gebruik [bestanden/MANIFEST.txt](bestanden/MANIFEST.txt) om te controleren
   welke artifacts aanwezig zijn.
5. Voor je presentatie: lees [PRESENTATIE.md](PRESENTATIE.md).

## Handleidingen per onderdeel

- [VM en projectmap maken](handleiding/01_vm_en_project/README.md)
- [Dependencies installeren](handleiding/02_dependencies/README.md)
- [Dataset voorbereiden](handleiding/03_dataset/README.md)
- [YOLO-detector trainen of klaarzetten](handleiding/04_detector/README.md)
- [Cropdataset en two-stage classifier trainen](handleiding/05_crop_classifier/README.md)
- [Exporteren en modelpakket maken](handleiding/06_export_modelpakket/README.md)
- [Controle, GitHub en grote bestanden](handleiding/07_controle_en_github/README.md)

## Snelle links naar belangrijke bestanden

- Finale cropdataset op Kaggle: [Smart Bin Classifier Crops](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-classifier-crops)
- Originele foto's op Kaggle: [Smart Bin Original Images](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-original-images)
- Huidige metadata: [two_stage_metadata.json](bestanden/huidig_modelpakket_test11mei_light_crops/two_stage_metadata.json)
- Huidige detector: [best.pt](bestanden/huidig_modelpakket_test11mei_light_crops/best.pt)
- Detector ONNX: [yolov8_detector.onnx](bestanden/huidig_modelpakket_test11mei_light_crops/yolov8_detector.onnx)
- Stage 1 ONNX: [stage1_main.onnx](bestanden/huidig_modelpakket_test11mei_light_crops/stage1_main.onnx)
- Stage 2 ONNX: [stage2_overige.onnx](bestanden/huidig_modelpakket_test11mei_light_crops/stage2_overige.onnx)
- Trainingsscript crops: [train_two_stage_crops.py](bestanden/scripts_en_config/train_two_stage_crops.py)
- Exportscript ONNX: [export_two_stage_onnx.py](bestanden/scripts_en_config/export_two_stage_onnx.py)
- Requirements: [requirements.txt](bestanden/scripts_en_config/requirements.txt)
- Presentatie-info: [PRESENTATIE.md](PRESENTATIE.md)

Opmerking: `yolov8_detector.onnx` is de bestandsnaam die in het huidige pakket
gebruikt wordt. In de uitleg noemen we dit gewoon de YOLO-detector. Er staan ook
YOLOv11-runs op de VM, maar de metadata van het huidige finale pakket verwijst
naar de detectorrun `garbage_detector_l_fallback_aware_768-6`.

## Belangrijke GitHub-opmerking

Sommige bestanden zijn groot. Vooral:

[bestanden/gebruikte_detector/best_detector_used_for_crops.pt](bestanden/gebruikte_detector/best_detector_used_for_crops.pt)

Dat bestand is ongeveer 350 MB en past niet in een normale GitHub commit. Gebruik
hiervoor Git LFS, een GitHub Release, of externe opslag met een downloadlink.

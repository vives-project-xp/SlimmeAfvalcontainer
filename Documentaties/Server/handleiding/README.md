# Handleiding vanaf nul

Deze handleiding is opgesplitst per onderdeel. Ze is bedoeld voor iemand die nog
geen VM, projectmap of trainingsomgeving heeft en het smart-bin trainingsresultaat
vanaf nul wil kunnen reproduceren.

## Volgorde

1. [VM en projectmap maken](01_vm_en_project/README.md)
2. [Dependencies installeren](02_dependencies/README.md)
3. [Dataset voorbereiden](03_dataset/README.md)
4. [YOLO-detector trainen of klaarzetten](04_detector/README.md)
5. [Cropdataset en two-stage classifier trainen](05_crop_classifier/README.md)
6. [Exporteren en modelpakket maken](06_export_modelpakket/README.md)
7. [Controle, GitHub en grote bestanden](07_controle_en_github/README.md)

## Wat wordt er uiteindelijk gemaakt?

Het doel is een map zoals:

```text
test11mei_light_crops/
|-- best.pt
|-- yolov8_detector.onnx
|-- stage1_main.onnx
|-- stage1_main.onnx.data
|-- stage2_overige.onnx
|-- stage2_overige.onnx.data
|-- two_stage_metadata.json
|-- classifier.py
|-- detector.py
|-- main.py
`-- pi_inference_two_stage.py
```

Een referentieversie van dat pakket staat in:

[../bestanden/huidig_modelpakket_test11mei_light_crops](../bestanden/huidig_modelpakket_test11mei_light_crops/)

## Belangrijke bijhorende bestanden

- [train_two_stage_crops.py](../bestanden/scripts_en_config/train_two_stage_crops.py)
- [export_two_stage_onnx.py](../bestanden/scripts_en_config/export_two_stage_onnx.py)
- [train_yolo_l.py](../bestanden/scripts_en_config/train_yolo_l.py)
- [train_yolo_until_target.py](../bestanden/scripts_en_config/train_yolo_until_target.py)
- [requirements.txt](../bestanden/scripts_en_config/requirements.txt)
- [two_stage_metadata.json](../bestanden/huidig_modelpakket_test11mei_light_crops/two_stage_metadata.json)

## Belangrijk uitgangspunt

Deze documentatie beschrijft serverkant training en export. De installatie op de
Raspberry Pi zelf hoort hier niet bij. De modelbestanden die later naar de Pi
gaan, worden hier wel gemaakt en verzameld.

# Handleiding Raspberry Pi 5 vanaf nul

Deze handleiding is opgesplitst per onderdeel. Ze is bedoeld voor iemand die de
Raspberry Pi 5-runtime van de slimme afvalcontainer opnieuw wil installeren of
controleren.

Training gebeurt niet op de Pi. De modellen worden serverkant getraind en daarna
als modelpakket naar de Pi gekopieerd.

## Volgorde

1. [Pi en projectmap klaarzetten](01_pi_en_projectmap/README.md)
2. [Dependencies installeren](02_dependencies/README.md)
3. [Modelpakket plaatsen](03_modelpakket/README.md)
4. [GUI en autostart instellen](04_gui_en_autostart/README.md)
5. [LEDs en ultrasoon controleren](05_leds_en_ultrasoon/README.md)
6. [Controle en troubleshooting](06_controle_en_troubleshooting/README.md)

## Detaildocumentatie

- [Two-stage deploygids referentie](03_modelpakket/referentie/deploy_gids_2_stage.md)
- [Hardwaredetails LEDs en ultrasoon](05_leds_en_ultrasoon/hardware/README.md)

## Wat draait er uiteindelijk?

Het doel is een automatisch startende GUI:

```text
garbagedetection-gui.service
`-- start_garbage_gui.sh
    `-- garbagedetection_gui.py --fullscreen
```

De GUI gebruikt dit modelpakket:

```text
finalmodel/
|-- yolov8_detector.onnx
|-- stage1_main.onnx
|-- stage1_main.onnx.data
|-- stage2_overige.onnx
|-- stage2_overige.onnx.data
|-- two_stage_metadata.json
|-- classifier.py
|-- detector.py
|-- led_controller.py
|-- ultrasone_controller.py
`-- garbagedetection_gui.py
```

## Belangrijke uitgangspunten

- De Pi draait de GUI, camera, LEDs en sensoren.
- De Pi traint geen modellen.
- Het modelpakket komt uit de serverflow.
- De huidige productieflow gebruikt `finalmodel`.
- De symlink `test11mei_light_crops -> finalmodel` houdt oude startpaden werkend.

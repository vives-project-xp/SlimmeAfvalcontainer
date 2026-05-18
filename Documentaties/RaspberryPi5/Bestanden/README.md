# Bestanden Raspberry Pi 5

Deze map beschrijft welke bestanden nodig zijn op de Raspberry Pi. De echte
runtimebestanden staan in de repo vooral onder:

```text
Code PI/
```

en op de Pi onder:

```text
/home/kobe/SlimmeAfvalcontainer/finalmodel
/home/kobe/SlimmeAfvalcontainer/Code PI
```

## Belangrijkste bestanden uit de repo

| Bestand | Doel |
|---|---|
| `Code PI/garbagedetection_gui.py` | Hoofd-GUI |
| `Code PI/led_controller.py` | LED-aansturing |
| `Code PI/ultrasone_controller.py` | Ultrasone sensoren |
| `Code PI/garbagedetection-gui.service` | systemd-servicevoorbeeld |
| `start_garbage_gui.sh` | startshellscript |
| `inference_gui.service` | ouder/alternatief servicebestand |

## Belangrijkste bestanden in het modelpakket

| Bestand | Doel |
|---|---|
| `yolov8_detector.onnx` | YOLO-detector voor objectdetectie |
| `stage1_main.onnx` | Stage 1 hoofdklasse-classifier |
| `stage1_main.onnx.data` | externe data voor Stage 1 ONNX |
| `stage2_overige.onnx` | Stage 2 classifier voor Overige |
| `stage2_overige.onnx.data` | externe data voor Stage 2 ONNX |
| `two_stage_metadata.json` | labels, modelnamen en metadata |
| `detector.py` | ONNX detector wrapper |
| `classifier.py` | ONNX/two-stage classifier wrapper |
| `garbagedetection_gui.py` | runtime GUI in het pakket |

## Opmerking over dubbele LED-controller

De GUI draait uit `finalmodel`, maar de LED-subprocess-code kan
`led_controller.py` uit `Code PI` importeren. Daarom moeten wijzigingen aan
LED-gedrag in beide bestanden gecontroleerd worden:

```text
/home/kobe/SlimmeAfvalcontainer/finalmodel/led_controller.py
/home/kobe/SlimmeAfvalcontainer/Code PI/led_controller.py
```


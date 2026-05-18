# 3. Modelpakket plaatsen

Het modelpakket komt uit de serverdocumentatie. De Pi gebruikt het pakket alleen
voor inference.

De oorspronkelijke korte deploygids staat nu als referentie hier:

[referentie/deploy_gids_2_stage.md](referentie/deploy_gids_2_stage.md)

## 3.1 Huidige locatie

Op de huidige Pi:

```text
/home/kobe/SlimmeAfvalcontainer/finalmodel
```

Controle:

```bash
ls -lah /home/kobe/SlimmeAfvalcontainer/finalmodel
```

## 3.2 Vereiste modelbestanden

Minimaal nodig:

```text
yolov8_detector.onnx
stage1_main.onnx
stage1_main.onnx.data
stage2_overige.onnx
stage2_overige.onnx.data
two_stage_metadata.json
```

Belangrijke Pythonbestanden:

```text
garbagedetection_gui.py
detector.py
classifier.py
led_controller.py
ultrasone_controller.py
```

## 3.3 Inference-flow

De runtime werkt in twee delen:

1. `yolov8_detector.onnx` zoekt objecten in het camerabeeld.
2. De beste crop wordt door de two-stage classifier gestuurd.

Two-stage classifier:

1. Stage 1 kiest een hoofdklasse:

```text
Organisch
PMD
Papier
Restafval
Overige
```

2. Alleen bij `Overige` draait Stage 2:

```text
Batterijen
Elektronica
Glas
Lightbulbs
Metaal
```

3. De GUI combineert dit als bijvoorbeeld:

```text
Overige/Batterijen
```

## 3.4 Metadata

Labels en thresholds mogen niet hardcoded worden als ze uit metadata kunnen
komen. Controleer:

```bash
cat /home/kobe/SlimmeAfvalcontainer/finalmodel/two_stage_metadata.json
```

Belangrijke velden:

```text
input_size
stage1_classes
stage2_overige_classes
stage1_model
stage2_model
main_label_for_stage2
default_fallback
```

## 3.5 Foto-opslag

De huidige runtime mag geen capturefoto's opslaan.

Controleer dat deze map niet terugkomt:

```bash
find /home/kobe/SlimmeAfvalcontainer/finalmodel -maxdepth 1 -type d -name captures
```

Controleer ook dat de GUI geen save-call meer uitvoert:

```bash
grep -n "_save_capture_photo(pil_img)" /home/kobe/SlimmeAfvalcontainer/finalmodel/garbagedetection_gui.py
```

Er hoort geen resultaat terug te komen.

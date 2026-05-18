# Test11mei Light Crops: Pi 5 + Hailo HAT pakket

Deze map is bedoeld als klaarzet-map voor de Raspberry Pi 5 met Hailo HAT.

Status:
- dit mapje is bedoeld voor de lichte detectorvariant voor Pi 5 + Hailo HAT
- de two-stage classifierbestanden zitten er al in
- die classifier is nu de crop-gebaseerde two-stage variant uit `models/two_stage_crops`
- de lichte detectorbestanden zitten er ook al in

Wat er nu al in zit:
- `classifier.py`
- `detector.py`
- `hailo_two_stage_main.py`
- `main.py`
- `pi_inference_two_stage.py`
- `stage1_main.onnx`
- `stage1_main.onnx.data`
- `stage2_overige.onnx`
- `stage2_overige.onnx.data`
- `two_stage_metadata.json`

Wat er nog ontbreekt voor Hailo:
- `yolov8_detector.hef`

## Verwachte detectorrun
Runmap:

```bash
/root/smart_bin_project/runs/yolo_light/garbage_detector_pi_light
```

## Stap 1: Controleer deze map
Deze bestanden zouden hier nu aanwezig moeten zijn:
- `best.pt`
- `yolov8_detector.onnx`
- `stage1_main.onnx`
- `stage2_overige.onnx`
- `two_stage_metadata.json`
- `hailo_two_stage_main.py`

## Stap 2: Detector opnieuw exporteren indien nodig
Als je later een nieuwere lichte detectorrun wilt inladen, voer je op de server uit:

```bash
cd /root/smart_bin_project/finalmodel
bash vul_detector_na_training.sh
```

Dat script doet:
- `best.pt` kopieren naar deze map
- `best.pt` exporteren naar ONNX
- de ONNX detector hernoemen naar `yolov8_detector.onnx`

Je kunt ook een andere runmap opgeven:

Standaard haalt dat script de detector uit:

```bash
/root/smart_bin_project/runs/yolo_light/garbage_detector_pi_light
```

## Stap 3: Kopieer deze map naar de Pi 5
Bijvoorbeeld:

```bash
scp -r /root/smart_bin_project/finalmodel pi@<IP_VAN_PI>:/home/pi/
```

## Stap 4: Detector klaarmaken voor Hailo
De Hailo runtime in `hailo_two_stage_main.py` verwacht een bestand met deze naam:

```bash
yolov8_detector.hef
```

Gebruik dus jullie normale Hailo compile-flow om van `best.pt` of `yolov8_detector.onnx` een `.hef` te maken en zet dat bestand in deze map met exact die naam.

Belangrijk:
- `main.py` gebruikt `yolov8_detector.onnx` voor CPU/ONNX-testen
- `hailo_two_stage_main.py` gebruikt `yolov8_detector.hef` voor de Hailo HAT

## Stap 5: Starten op de Pi
Voor Hailo HAT:

```bash
cd ~/finalmodel
python3 hailo_two_stage_main.py
```

Voor ONNX/CPU test:

```bash
cd ~/finalmodel
python3 main.py
```

## Stap 6: Snelle visuele controle
Na training kun je hier evaluatiebestanden verwachten in `evaluatie/`.

## Opmerking
De two-stage classifier in deze map is nu de crop-gebaseerde variant uit `models/two_stage_crops`:
- `stage1_main` validatie-accuracy: `89.46%`
- `stage2_overige` validatie-accuracy: `92.03%`

De detector kan later nog vervangen worden door een nieuwere `best.pt`/`yolov8_detector.onnx`/`.hef` zonder dat je de classifierbestanden opnieuw hoeft te maken.


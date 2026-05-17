# 4. YOLO-detector trainen of klaarzetten

De detector zoekt het afvalobject in de foto. Daarna kan de classifier op een crop trainen in plaats van op de volledige afbeelding.

Korte uitleg:

- `YOLO`: het detectiemodel dat objecten en bounding boxes vindt.
- `best.pt`: het beste detectorbestand uit een trainingsrun.
- `mAP50`: score die aangeeft hoe goed de detector objecten vindt.

Opmerking: op de VM bestaan ook YOLOv11-runs. De metadata van het huidige finale pakket verwijst echter naar de detectorrun `garbage_detector_l_fallback_aware_768-6`. In het modelpakket heet de export `yolov8_detector.onnx`; dat is een legacy bestandsnaam die de code verwacht. Daarom spreken we in de handleiding meestal over de YOLO-detector in het algemeen.

## 4.1 Keuze: bestaande detector of nieuwe training

Voor exacte reproductie gebruik je de bestaande detector uit:

- [Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)
- [Gebruikte_detector/README.md](../../Bestanden/Gebruikte_detector/README.md)

Origineel serverpad:

```text
/root/smart_bin_project/runs/detect_strong/garbage_detector_l_fallback_aware_768-6/weights/best.pt
```

Als je echt vanaf nul werkt, moet je deze detector eerst opnieuw trainen.

## 4.2 Scripts

Beschikbare scripts:

- [train_yolo_l.py](../../Bestanden/Scripts_en_config/train_yolo_l.py)
- [train_yolo_until_target.py](../../Bestanden/Scripts_en_config/train_yolo_until_target.py)

`train_yolo_l.py` is eenvoudiger. `train_yolo_until_target.py` is uitgebreider en kan meerdere pogingen doen tot een doelmetric gehaald wordt.

## 4.3 Dataset voorbereiden voor YOLO

YOLO heeft een train/val lijst en YAML nodig:

```text
yolo_train.txt
yolo_val.txt
dataset.yaml
```

## 4.4 Eenvoudige training

```bash
cd /root/smart_bin_project
source .venv/bin/activate
python3 train_yolo_l.py
```

Belangrijke instellingen uit het script:

```text
epochs=80
imgsz=640
batch=16
device=0
patience=15
```

Wat betekent dit kort?

- `epochs`: hoe vaak het model door de trainingsdata gaat.
- `imgsz`: beeldgrootte waarmee YOLO traint.
- `batch`: hoeveel beelden tegelijk door de GPU gaan.
- `device=0`: gebruik GPU 0.
- `patience`: stop als er zoveel epochs geen verbetering is.

## 4.5 Detector exporteren naar ONNX

Voor de detector die in het huidige model gebruikt werd:

```bash
yolo export model=/root/smart_bin_project/runs/detect_strong/garbage_detector_l_fallback_aware_768-6/weights/best.pt format=onnx imgsz=768
```

Het exportbestand heet meestal:

```text
best.onnx
```

Voor het modelpakket moet dit worden:

```text
yolov8_detector.onnx
```

Laat die naam staan tenzij je ook de code aanpast die dit bestand inlaadt.

Referentie:

[yolov8_detector.onnx](../../Bestanden/Huidig_modelpakket_test11mei_light_crops/yolov8_detector.onnx)

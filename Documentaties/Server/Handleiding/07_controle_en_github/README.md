# 7. Controle, GitHub en grote bestanden

Dit onderdeel gaat over controleren of alles compleet is en hoe je dit logisch
in GitHub bewaart.

## 7.1 Controleer modelpakket

```bash
ls -lh /root/smart_bin_project/test_reproductie_light_crops
```

Je verwacht minstens:

```text
best.pt
yolov8_detector.onnx
stage1_main.onnx
stage1_main.onnx.data
stage2_overige.onnx
stage2_overige.onnx.data
two_stage_metadata.json
classifier.py
detector.py
main.py
pi_inference_two_stage.py
```

Vergelijk met de referentie:

[Huidig_modelpakket_test11mei_light_crops](../../Bestanden/Huidig_modelpakket_test11mei_light_crops/)

## 7.2 Manifest gebruiken

Gebruik:

- [MANIFEST.txt](../../Bestanden/MANIFEST.txt)
- [LOCAL_MANIFEST.txt](../../Bestanden/LOCAL_MANIFEST.txt)

`MANIFEST.txt` is de serverlijst. `LOCAL_MANIFEST.txt` toont wat lokaal in deze
workspace aanwezig was toen de documentatie werd gemaakt.

## 7.3 Metadata controleren

Open:

[two_stage_metadata.json](../../Bestanden/Huidig_modelpakket_test11mei_light_crops/two_stage_metadata.json)

Controleer:

```text
input_size = 224
stage1_classes bevat 5 hoofdklassen
stage2_overige_classes bevat 5 Overige-subklassen
detector_used wijst naar de juiste detector
crops_root wijst naar de juiste cropdataset
```

## 7.4 GitHub-structuur

Aanbevolen:

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
    |-- Scripts_en_config/
    |-- Huidig_modelpakket_test11mei_light_crops/
    |-- two_stage_training_outputs/
    `-- Gebruikte_detector/
```

## 7.5 Grote bestanden

Gewone GitHub commits hebben een harde limiet rond 100 MB per bestand. Dit is
een probleem voor:

[best_detector_used_for_crops.pt](../../Bestanden/Gebruikte_detector/best_detector_used_for_crops.pt)

Dit bestand is ongeveer 350 MB.

Gebruik een van deze oplossingen:

```text
Git LFS
GitHub Release assets
externe opslag met downloadlink
serverpad behouden en documenteren
```

## 7.6 Wat wel makkelijk in GitHub kan

Deze bestanden zijn veel geschikter om gewoon mee te nemen:

- [train_two_stage_crops.py](../../Bestanden/Scripts_en_config/train_two_stage_crops.py)
- [export_two_stage_onnx.py](../../Bestanden/Scripts_en_config/export_two_stage_onnx.py)
- [requirements.txt](../../Bestanden/Scripts_en_config/requirements.txt)
- [dataset.yaml](../../Bestanden/Scripts_en_config/dataset.yaml)
- [two_stage_metadata.json](../../Bestanden/Huidig_modelpakket_test11mei_light_crops/two_stage_metadata.json)

Ook kleinere ONNX/modelpakketbestanden kunnen afhankelijk van jullie repo-afspraken
mee, maar check altijd de bestandsgrootte.

## 7.7 Laatste sanity check

Op server:

```bash
find /root/smart_bin_project/documentatie -maxdepth 3 -type f | sort
du -sh /root/smart_bin_project/documentatie
df -h
```

Als de disk vol staat, maak geen extra kopieën van grote modellen. Gebruik dan
hardlinks, Git LFS of externe artifacts.


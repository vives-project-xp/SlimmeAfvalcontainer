# 7. Controle, GitHub en grote bestanden

Dit onderdeel gaat over controleren of alles compleet is en hoe je dit logisch in GitHub bewaart.

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

[Huidig_modelpakket_finalmodel](../../Bestanden/Huidig_modelpakket_finalmodel/)

## 7.2 Manifest gebruiken

Gebruik:

- [MANIFEST.txt](../../Bestanden/MANIFEST.txt)
- [LOCAL_MANIFEST.txt](../../Bestanden/LOCAL_MANIFEST.txt)

`MANIFEST.txt` is de serverlijst. `LOCAL_MANIFEST.txt` toont wat bewust in GitHub staat en wat via Kaggle gedownload moet worden.

## 7.3 Metadata controleren

Open:

[two_stage_metadata.json](../../Bestanden/Huidig_modelpakket_finalmodel/two_stage_metadata.json)

Controleer:

```text
input_size = 224
stage1_classes bevat 5 hoofdklassen
stage2_overige_classes bevat 5 Overige-subklassen
detector_used wijst naar de juiste detector
crops_root wijst naar de juiste cropdataset
```

## 7.4 Aanbevolen verdeling

Gebruik deze structuur:

```text
GitHub:
- code
- documentatie
- scripts en configs
- compact referentie-modelpakket

Kaggle:
- Smart Bin Original Images
- Smart Bin Classifier Crops
- Smart Bin Model Artifacts
```

Links:

- [Smart Bin Original Images](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-original-images)
- [Smart Bin Classifier Crops](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-classifier-crops)
- [Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)

## 7.5 Grote bestanden

De zwaarste bestanden, zoals de detector voor cropbouw en de volledige trainingoutputs, staan bewust niet als gewone binaries in GitHub.

Gebruik daarom de Kaggle model-artifacts als vaste downloadplek:

- [Gebruikte_detector/README.md](../../Bestanden/Gebruikte_detector/README.md)
- [two_stage_training_outputs/README.md](../../Bestanden/two_stage_training_outputs/README.md)

## 7.6 Wat wel makkelijk in GitHub kan

Deze bestanden zijn veel geschikter om gewoon mee te nemen:

- [train_two_stage_crops.py](../../Bestanden/Scripts_en_config/train_two_stage_crops.py)
- [export_two_stage_onnx.py](../../Bestanden/Scripts_en_config/export_two_stage_onnx.py)
- [requirements.txt](../../Bestanden/Scripts_en_config/requirements.txt)
- [dataset.yaml](../../Bestanden/Scripts_en_config/dataset.yaml)
- [two_stage_metadata.json](../../Bestanden/Huidig_modelpakket_finalmodel/two_stage_metadata.json)

Ook kleinere ONNX/modelpakketbestanden kunnen mee, zolang de bestandsgrootte werkbaar blijft.

## 7.7 Laatste sanity check

Als je alles opnieuw wil opbouwen, heb je nodig:

1. deze GitHub-repo
2. [Smart Bin Original Images](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-original-images)
3. [Smart Bin Classifier Crops](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-classifier-crops)
4. [Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)

Met die combinatie heb je de documentatie, scripts, datasets, detector en trainingsoutputs samen.


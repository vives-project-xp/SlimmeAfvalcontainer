# 5. Cropdataset en two-stage classifier trainen

Dit onderdeel maakt het belangrijkste classificatiemodel.

## 5.1 Waarom crops?

Als je een volledige foto classificeert, kan het model te veel leren van de achtergrond. Daarom wordt eerst met YOLO het afvalobject gezocht. Van dat object wordt een crop gemaakt. De classifier traint daarna op die crops.

Voor het finale model gebruiken we deze cropdataset:

[Smart Bin Classifier Crops](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-classifier-crops)

Deze crops zijn gemaakt uit de originele foto-dataset:

[Smart Bin Original Images](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-original-images)

De originele foto's kwamen uit datasets die via Google Dataset Search en Kaggle gevonden zijn, aangevuld met zoekwerk via [images.cv](https://images.cv/). Omdat een deel van de beelden uit Amerikaanse bronnen kwam, moesten we zelf sorteren en controleren. Vooral `PMD` vroeg extra aandacht, omdat de PMD-regels in Brugge niet hetzelfde zijn als de Amerikaanse sorteerlogica.

Korte uitleg:

- `crop`: uitgesneden deel van een foto.
- `classifier`: model dat een crop in een afvalklasse indeelt.
- `two-stage`: eerst hoofdklasse voorspellen, daarna eventueel subklasse.

## 5.2 Script

Gebruik:

[train_two_stage_crops.py](../../Bestanden/Scripts_en_config/train_two_stage_crops.py)

## 5.3 Belangrijke paden in het script

```text
SOURCE_ROOT = /root/smart_bin_project/data/Dataset/train
BEST_DETECTOR = /root/smart_bin_project/runs/detect_strong/garbage_detector_l_fallback_aware_768-6/weights/best.pt
CROPS_ROOT = /root/smart_bin_project/data/classifier_crops
OUTPUT_DIR = /root/smart_bin_project/models/two_stage_crops
LOG_PATH = /root/smart_bin_project/two_stage_crops_training.log
```

Als je een andere detectorrun gebruikt, pas `BEST_DETECTOR` aan.

## 5.4 Cropdataset bouwen

De cropdataset wordt automatisch gebouwd bij het starten van:

```bash
cd /root/smart_bin_project
source .venv/bin/activate
python3 train_two_stage_crops.py
```

Output:

```text
/root/smart_bin_project/data/classifier_crops/train
/root/smart_bin_project/data/classifier_crops/val
```

Op Kaggle staat deze finale cropdataset hier:

[Smart Bin Classifier Crops](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-classifier-crops)

## 5.5 Two-stage classifier

Het script traint twee MobileNetV3 Large modellen:

```text
Stage 1: Organisch, PMD, Papier, Restafval, Overige
Stage 2: Batterijen, Elektronica, Glas, Lightbulbs, Metaal
```

Stage 2 wordt alleen gebruikt voor afbeeldingen die Stage 1 als `Overige` inschat.

## 5.6 Hyperparameters

Standaard:

```text
EPOCHS_STAGE1 = 50
EPOCHS_STAGE2 = 50
BATCH_SIZE = 64
NUM_WORKERS = 8
LEARNING_RATE = 0.001
IMG_SIZE = 224
SEED = 42
VRAM_LIMIT_MB = 18432
```

Wat betekent dit kort?

- `EPOCHS_STAGE1` en `EPOCHS_STAGE2`: hoe lang beide modellen trainen.
- `BATCH_SIZE`: hoeveel beelden tegelijk door het model gaan.
- `NUM_WORKERS`: hoeveel parallelle dataloading-processen gebruikt worden.
- `LEARNING_RATE`: hoe groot de leerstappen zijn.
- `IMG_SIZE`: de inputgrootte van de classifier.
- `SEED`: vaste random startwaarde voor reproduceerbaarheid.
- `VRAM_LIMIT_MB`: maximum GPU-geheugen dat het script probeert te gebruiken.

## 5.7 Output

Na training krijg je:

- `stage1_main.pth`
- `stage2_overige.pth`
- `stage1_main.onnx`
- `stage1_main.onnx.data`
- `stage2_overige.onnx`
- `stage2_overige.onnx.data`
- `two_stage_metadata.json`

Downloadbare referentieversie:

- [Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)
- [two_stage_training_outputs/README.md](../../Bestanden/two_stage_training_outputs/README.md)

Op de server:

```text
/root/smart_bin_project/models/two_stage_crops
```

## 5.8 Verwachte metadata

De metadata bevat onder andere:

```json
{
  "input_size": 224,
  "stage1_classes": ["Organisch", "PMD", "Papier", "Restafval", "Overige"],
  "stage2_overige_classes": ["Batterijen", "Elektronica", "Glas", "Lightbulbs", "Metaal"],
  "main_label_for_stage2": "Overige",
  "default_fallback": "Restafval"
}
```

Referentie:

[two_stage_metadata.json](../../Bestanden/Huidig_modelpakket_test11mei_light_crops/two_stage_metadata.json)

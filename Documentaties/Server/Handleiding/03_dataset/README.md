# 3. Dataset voorbereiden

Dit onderdeel legt uit welke data nodig is en hoe de mappen moeten staan.

## 3.1 Verwachte locatie

De dataset moet hier komen:

```text
/root/smart_bin_project/data/Dataset
```

De belangrijkste map voor training:

```text
/root/smart_bin_project/data/Dataset/train
```

## 3.1.1 Bron van de foto's

De foto's zijn verzameld via meerdere bronnen:

- Kaggle datasets, gevonden via Google Dataset Search.
- Extra beeldmateriaal/zoekwerk via [images.cv](https://images.cv/).
- Eigen sorteerwerk om de klassen bruikbaar te maken voor de afvalregels in Brugge.

Belangrijk: sommige foto's kwamen uit Amerikaanse datasets. Daardoor kloppen de afvalcategorieen niet altijd met de sorteerregels in Brugge. Vooral bij `PMD` moesten we zelf foto's controleren en sorteren, omdat PMD hier anders werkt dan in Amerika.

Voor deze documentatie gebruiken en vermelden we de datasets die we zelf hebben klaargezet op Kaggle:

- originele/normale foto's: [Smart Bin Original Images](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-original-images)
- crops voor het finale model: [Smart Bin Classifier Crops](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-classifier-crops)
- zware modelbestanden en trainingsoutputs: [Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)

Op de VM staat de originele dataset uitgepakt onder:

```text
/root/smart_bin_project/data/Dataset
```

Voor het finale model gebruiken we niet rechtstreeks de volledige originele foto's, maar de cropdataset die daaruit gemaakt is:

[Smart Bin Classifier Crops](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-classifier-crops)

Die cropdataset komt dus voort uit de originele foto's. Eerst zoekt YOLO het afvalobject in de originele foto, daarna wordt dat object uitgesneden als crop. Die crops zijn gebruikt om het finale two-stage classifiermodel te trainen.

## 3.2 Verwachte structuur

```text
data/Dataset/train/
|-- Organisch/
|-- PMD/
|-- Papier/
|-- Restafval/
`-- Overige/
    |-- Batterijen/
    |-- Elektronica/
    |-- Glas/
    |-- Lightbulbs/
    `-- Metaal/
```

De hoofdletters zijn belangrijk. De scripts gebruiken deze namen.

## 3.3 Data kopieren naar VM

Als je lokaal een datasetmap hebt:

```bash
scp -r Dataset root@<IP>:/root/smart_bin_project/data/
```

Of download rechtstreeks van Kaggle en pak uit op de VM.

Controle:

```bash
ls /root/smart_bin_project/data/Dataset/train
```

## 3.4 Klassen

Hoofdklassen:

```text
Organisch
PMD
Papier
Restafval
Overige
```

Subklassen voor Stage 2:

```text
Batterijen
Elektronica
Glas
Lightbulbs
Metaal
```

## 3.5 YOLO-labels

YOLO gebruikt `.txt` bestanden naast de afbeeldingen:

```text
foto_001.jpg
foto_001.txt
```

Inhoud:

```text
class_id center_x center_y width height
```

Alle waarden behalve `class_id` zijn genormaliseerd tussen 0 en 1.

Korte uitleg:

- `class_id`: nummer van de afvalklasse.
- `center_x` en `center_y`: middelpunt van de box.
- `width` en `height`: breedte en hoogte van de box.
- Genormaliseerd betekent dat de waarde tussen 0 en 1 staat in plaats van in pixels.

Mapping voor detectortraining:

```text
0 Organisch
1 PMD
2 Papier
3 Restafval
4 Overige
```

Alle submappen onder `Overige` worden voor YOLO dus class id `4`. De subklasse-herkenning gebeurt later met de two-stage classifier.

## 3.6 Dataset YAML

Voor YOLO heb je een YAML nodig. Voorbeelden:

- [dataset.yaml](../../Bestanden/Scripts_en_config/dataset.yaml)
- [dataset_extended.yaml](../../Bestanden/Scripts_en_config/dataset_extended.yaml)
- [yolov11x_data.yaml](../../Bestanden/Scripts_en_config/yolov11x_data.yaml)

Een eenvoudige `dataset.yaml` ziet er zo uit:

```yaml
path: /root/smart_bin_project
train: yolo_train.txt
val: yolo_val.txt
names:
  0: Organisch
  1: PMD
  2: Papier
  3: Restafval
  4: Overige
```

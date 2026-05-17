# Bestanden

Deze map bevat de referentiebestanden die horen bij het huidige trainingsresultaat.

De README's in de andere mappen leggen uit hoe je ze gebruikt en opnieuw maakt.

## Structuur

```text
Bestanden/
|-- README.md
|-- MANIFEST.txt
|-- LOCAL_MANIFEST.txt
|-- Huidig_modelpakket_test11mei_light_crops/
|-- two_stage_training_outputs/
|-- Gebruikte_detector/
`-- Scripts_en_config/
```

## Wat staat lokaal in GitHub?

In deze repo houden we vooral bij:

- het compacte huidige modelpakket
- scripts en configuratie
- manifesten en documentatie

De zwaardere trainingsbestanden staan bewust op Kaggle:

[Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)

## `Huidig_modelpakket_test11mei_light_crops`

Dit is het huidige modelpakket dat als referentie gebruikt wordt.

Belangrijkste bestanden:

- [best.pt](Huidig_modelpakket_test11mei_light_crops/best.pt)
- [yolov8_detector.onnx](Huidig_modelpakket_test11mei_light_crops/yolov8_detector.onnx)
- [stage1_main.onnx](Huidig_modelpakket_test11mei_light_crops/stage1_main.onnx)
- [stage1_main.onnx.data](Huidig_modelpakket_test11mei_light_crops/stage1_main.onnx.data)
- [stage2_overige.onnx](Huidig_modelpakket_test11mei_light_crops/stage2_overige.onnx)
- [stage2_overige.onnx.data](Huidig_modelpakket_test11mei_light_crops/stage2_overige.onnx.data)
- [two_stage_metadata.json](Huidig_modelpakket_test11mei_light_crops/two_stage_metadata.json)
- [classifier.py](Huidig_modelpakket_test11mei_light_crops/classifier.py)
- [detector.py](Huidig_modelpakket_test11mei_light_crops/detector.py)
- [main.py](Huidig_modelpakket_test11mei_light_crops/main.py)
- [pi_inference_two_stage.py](Huidig_modelpakket_test11mei_light_crops/pi_inference_two_stage.py)
- [hailo_two_stage_main.py](Huidig_modelpakket_test11mei_light_crops/hailo_two_stage_main.py)
- [vul_detector_na_training.sh](Huidig_modelpakket_test11mei_light_crops/vul_detector_na_training.sh)
- [LEES_MIJ_PI5_HAILO_TEST11MEI_LIGHT_CROPS.md](Huidig_modelpakket_test11mei_light_crops/LEES_MIJ_PI5_HAILO_TEST11MEI_LIGHT_CROPS.md)

Dit komt overeen met de servermap:

```text
/root/smart_bin_project/test11mei_light_crops
```

Opmerking: `yolov8_detector.onnx` is een bestaande bestandsnaam in het pakket.
Voor de documentatie noemen we dit de YOLO-detector export. Hernoem dit bestand
niet zomaar, want scripts kunnen exact deze naam verwachten.

## `two_stage_training_outputs`

Deze map bevat in GitHub alleen uitleg en verwijzingen.

De echte trainingsoutputs staan op Kaggle:

- [Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)
- [two_stage_training_outputs/README.md](two_stage_training_outputs/README.md)

Die komen van:

```text
/root/smart_bin_project/models/two_stage_crops
```

## `Gebruikte_detector`

Deze map bevat in GitHub alleen uitleg en verwijzingen.

De detector die gebruikt werd om de cropdataset te bouwen staat op Kaggle:

- [Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)
- [Gebruikte_detector/README.md](Gebruikte_detector/README.md)

Origineel serverpad:

```text
/root/smart_bin_project/runs/detect_strong/garbage_detector_l_fallback_aware_768-6/weights/best.pt
```

## `Scripts_en_config`

Deze map bevat de scripts en configuratiebestanden die nodig zijn om training en export opnieuw te doen.

Belangrijkste bestanden:

- [train_two_stage_crops.py](Scripts_en_config/train_two_stage_crops.py)
- [export_two_stage_onnx.py](Scripts_en_config/export_two_stage_onnx.py)
- [train_two_stage.py](Scripts_en_config/train_two_stage.py)
- [train_yolo_l.py](Scripts_en_config/train_yolo_l.py)
- [train_yolo_until_target.py](Scripts_en_config/train_yolo_until_target.py)
- [verzamel_pi_bestanden.sh](Scripts_en_config/verzamel_pi_bestanden.sh)
- [requirements.txt](Scripts_en_config/requirements.txt)
- [dataset.yaml](Scripts_en_config/dataset.yaml)
- [dataset_extended.yaml](Scripts_en_config/dataset_extended.yaml)
- [yolov11x_data.yaml](Scripts_en_config/yolov11x_data.yaml)

## Manifest

[MANIFEST.txt](MANIFEST.txt) bevat een lijst van alle bestanden in deze map zoals ze op de server geplaatst zijn.

[LOCAL_MANIFEST.txt](LOCAL_MANIFEST.txt) beschrijft welke bestanden bewust lokaal in GitHub staan en welke via Kaggle gedownload moeten worden.

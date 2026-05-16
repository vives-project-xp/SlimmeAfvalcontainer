# Bestanden

Deze map bevat de echte bestanden die horen bij het huidige trainingsresultaat.
De README's in de andere mappen leggen uit hoe je ze gebruikt en opnieuw maakt.

## Structuur

```text
bestanden/
|-- README.md
|-- MANIFEST.txt
|-- huidig_modelpakket_test11mei_light_crops/
|-- two_stage_training_outputs/
|-- gebruikte_detector/
`-- scripts_en_config/
```

## `huidig_modelpakket_test11mei_light_crops`

Dit is het huidige modelpakket dat als referentie gebruikt wordt.

Belangrijkste bestanden:

- [best.pt](huidig_modelpakket_test11mei_light_crops/best.pt)
- [yolov8_detector.onnx](huidig_modelpakket_test11mei_light_crops/yolov8_detector.onnx)
- [stage1_main.onnx](huidig_modelpakket_test11mei_light_crops/stage1_main.onnx)
- [stage1_main.onnx.data](huidig_modelpakket_test11mei_light_crops/stage1_main.onnx.data)
- [stage2_overige.onnx](huidig_modelpakket_test11mei_light_crops/stage2_overige.onnx)
- [stage2_overige.onnx.data](huidig_modelpakket_test11mei_light_crops/stage2_overige.onnx.data)
- [two_stage_metadata.json](huidig_modelpakket_test11mei_light_crops/two_stage_metadata.json)
- [classifier.py](huidig_modelpakket_test11mei_light_crops/classifier.py)
- [detector.py](huidig_modelpakket_test11mei_light_crops/detector.py)
- [main.py](huidig_modelpakket_test11mei_light_crops/main.py)
- [pi_inference_two_stage.py](huidig_modelpakket_test11mei_light_crops/pi_inference_two_stage.py)
- [hailo_two_stage_main.py](huidig_modelpakket_test11mei_light_crops/hailo_two_stage_main.py)
- [vul_detector_na_training.sh](huidig_modelpakket_test11mei_light_crops/vul_detector_na_training.sh)
- [LEES_MIJ_PI5_HAILO_TEST11MEI_LIGHT_CROPS.md](huidig_modelpakket_test11mei_light_crops/LEES_MIJ_PI5_HAILO_TEST11MEI_LIGHT_CROPS.md)

Dit komt overeen met de servermap:

```text
/root/smart_bin_project/test11mei_light_crops
```

Opmerking: `yolov8_detector.onnx` is een bestaande bestandsnaam in het pakket.
Voor de documentatie noemen we dit de YOLO-detector export. Hernoem dit bestand
niet zomaar, want scripts kunnen exact deze naam verwachten.

## `two_stage_training_outputs`

Dit zijn de outputs van de two-stage classifiertraining.

Belangrijkste bestanden:

- [stage1_main.pth](two_stage_training_outputs/stage1_main.pth)
- [stage2_overige.pth](two_stage_training_outputs/stage2_overige.pth)
- [stage1_main.onnx](two_stage_training_outputs/stage1_main.onnx)
- [stage1_main.onnx.data](two_stage_training_outputs/stage1_main.onnx.data)
- [stage2_overige.onnx](two_stage_training_outputs/stage2_overige.onnx)
- [stage2_overige.onnx.data](two_stage_training_outputs/stage2_overige.onnx.data)
- [two_stage_metadata.json](two_stage_training_outputs/two_stage_metadata.json)

Deze komen van:

```text
/root/smart_bin_project/models/two_stage_crops
```

## `gebruikte_detector`

Deze map bevat de detector die gebruikt werd om de cropdataset te bouwen:

- [best_detector_used_for_crops.pt](gebruikte_detector/best_detector_used_for_crops.pt)

Origineel serverpad:

```text
/root/smart_bin_project/runs/detect_strong/garbage_detector_l_fallback_aware_768-6/weights/best.pt
```

Let op: dit bestand is ongeveer 350 MB. Voor GitHub moet dit via Git LFS,
GitHub Releases of externe opslag. Een normale GitHub commit accepteert geen
bestanden boven 100 MB.

## `scripts_en_config`

Deze map bevat de scripts en configuratiebestanden die nodig zijn om training en
export opnieuw te doen.

Belangrijkste bestanden:

- [train_two_stage_crops.py](scripts_en_config/train_two_stage_crops.py)
- [export_two_stage_onnx.py](scripts_en_config/export_two_stage_onnx.py)
- [train_two_stage.py](scripts_en_config/train_two_stage.py)
- [train_yolo_l.py](scripts_en_config/train_yolo_l.py)
- [train_yolo_until_target.py](scripts_en_config/train_yolo_until_target.py)
- [verzamel_pi_bestanden.sh](scripts_en_config/verzamel_pi_bestanden.sh)
- [requirements.txt](scripts_en_config/requirements.txt)
- [dataset.yaml](scripts_en_config/dataset.yaml)
- [dataset_extended.yaml](scripts_en_config/dataset_extended.yaml)
- [yolov11x_data.yaml](scripts_en_config/yolov11x_data.yaml)

## Manifest

[MANIFEST.txt](MANIFEST.txt) bevat een lijst van alle bestanden in deze map zoals
ze op de server geplaatst zijn.

[LOCAL_MANIFEST.txt](LOCAL_MANIFEST.txt) bevat de lokale status van de bestanden
in deze workspace.

Gebruik dit om te controleren of een lokale of GitHub-kopie compleet is.

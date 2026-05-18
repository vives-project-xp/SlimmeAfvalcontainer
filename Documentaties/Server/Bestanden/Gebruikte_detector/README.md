# Gebruikte detector

Deze map bevat in GitHub geen zware `.pt` binary, maar wel de uitleg over welke detector gebruikt werd.

## Bestand

De detector die gebruikt werd om de cropdataset te bouwen is:

- `best_detector_used_for_crops.pt`

## Download

Download dit bestand via:

[Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)

In die Kaggle dataset zit een tarbestand `used_crop_detector.tar` met daarin:

```text
used_crop_detector/
`-- best_detector_used_for_crops.pt
```

## Origineel serverpad

```text
/root/smart_bin_project/runs/detect_strong/garbage_detector_l_fallback_aware_768-6/weights/best.pt
```

## Opmerking

Deze detector is gebruikt om de objectcrops uit de originele foto's te maken. Het finale modelpakket bevat daarnaast ook een compactere `best.pt` in:

```text
/root/smart_bin_project/finalmodel
```

Voor exacte reproductie van de cropbouw gebruik je de detector uit de Kaggle model-artifacts.


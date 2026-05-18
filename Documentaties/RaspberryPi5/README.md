# Smart Bin Raspberry Pi 5 documentatie

Deze map bevat de documentatie om het Raspberry Pi-gedeelte van de slimme
afvalcontainer te begrijpen, opnieuw klaar te zetten en te controleren.

Het gaat hier over de runtime op de Pi: GUI, camera, modelpakket, LED-strips,
ultrasone sensoren en autostart. Training en export van de modellen staan in:

[../Server](../Server/README.md)

## Inhoud

```text
RaspberryPi5/
|-- README.md
|-- Handleiding/
|   |-- README.md
|   |-- 01_pi_en_projectmap/
|   |-- 02_dependencies/
|   |-- 03_modelpakket/
|   |-- 04_gui_en_autostart/
|   |-- 05_leds_en_ultrasoon/
|   `-- 06_controle_en_troubleshooting/
`-- Bestanden/
    |-- README.md
    `-- MANIFEST.txt
```

## Waarvoor dient deze map?

- [Handleiding/](Handleiding/README.md) legt stap voor stap uit hoe de Pi wordt
  klaargezet en hoe de GUI automatisch start.
- [Bestanden/](Bestanden/README.md) beschrijft welke bestanden uit de repo en
  het modelpakket op de Pi nodig zijn.
- [Handleiding/05_leds_en_ultrasoon/hardware/](Handleiding/05_leds_en_ultrasoon/hardware/)
  bevat de detaildocumentatie voor LED-strips, ultrasone sensoren en bedrading.
- [Handleiding/03_modelpakket/referentie/deploy_gids_2_stage.md](Handleiding/03_modelpakket/referentie/deploy_gids_2_stage.md)
  bewaart de oorspronkelijke korte deploygids voor het two-stage model.

## Huidige referentie-installatie op de Pi

De actuele Pi gebruikt deze projectmap:

```text
/home/kobe/SlimmeAfvalcontainer
```

De GUI-service start via:

```text
/etc/systemd/system/garbagedetection-gui.service
```

Die service voert dit script uit:

```text
/home/kobe/SlimmeAfvalcontainer/start_garbage_gui.sh
```

Het huidige modelpakket staat op de Pi onder:

```text
/home/kobe/SlimmeAfvalcontainer/finalmodel
```

Voor compatibiliteit bestaat er ook een symlink:

```text
/home/kobe/SlimmeAfvalcontainer/test11mei_light_crops -> finalmodel
```

Die symlink is belangrijk zolang bestaande startbestanden nog naar
`test11mei_light_crops` verwijzen.

## Snel starten

1. Lees [Handleiding/README.md](Handleiding/README.md).
2. Begin met [01_pi_en_projectmap](Handleiding/01_pi_en_projectmap/README.md).
3. Installeer dependencies via [02_dependencies](Handleiding/02_dependencies/README.md).
4. Plaats of controleer het modelpakket via [03_modelpakket](Handleiding/03_modelpakket/README.md).
5. Zet autostart klaar via [04_gui_en_autostart](Handleiding/04_gui_en_autostart/README.md).
6. Controleer LEDs en ultrasoon via [05_leds_en_ultrasoon](Handleiding/05_leds_en_ultrasoon/README.md).
7. Gebruik [06_controle_en_troubleshooting](Handleiding/06_controle_en_troubleshooting/README.md) bij problemen.

## Belangrijkste runtimebestanden

- `garbagedetection_gui.py`: hoofd-GUI met camera, detectie, classificatie en UI.
- `detector.py`: YOLO ONNX-detectie.
- `classifier.py`: two-stage ONNX-classificatie.
- `led_controller.py`: aansturing van de vier WS2812B LED-strips.
- `ultrasone_controller.py`: monitoring van de vier ultrasone sensoren.
- `start_garbage_gui.sh`: startshellscript voor systemd.
- `garbagedetection-gui.service`: systemd-service voor automatische start.

## Belangrijk gedrag

De Pi gebruikt een detector + classifier-flow:

1. YOLO ONNX zoekt een object in het camerabeeld.
2. De beste crop wordt door de two-stage classifier gestuurd.
3. Stage 1 kiest de hoofdklasse.
4. Bij `Overige` draait Stage 2 voor de subklasse.
5. De GUI vertaalt het resultaat naar een bak of reject-status.
6. LEDs tonen de gekozen bak of een rood reject-signaal.
7. Ultrasone sensoren bevestigen inworp en tonen vulgraad.

## GitHub-opmerking

Niet pushen zonder eerst te controleren welke grote modelbestanden in Git horen.
Voor training en zware artifacts wordt dezelfde verdeling gevolgd als in de
serverdocumentatie:

- GitHub voor code, documentatie, scripts en compacte referentiebestanden
- externe opslag voor zware datasets en grote trainingsoutputs

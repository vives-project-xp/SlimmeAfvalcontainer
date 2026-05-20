# Slimme Afvalcontainer

Een intelligente afvalcontainer met camera-gebaseerde objectdetectie, LED-indicatie en ultrasoon sensordetectie. 

## Inhoudstafel

| Onderdeel | Korte uitleg |
|---|---|
| [`Architectuur/`](Architectuur/) | Schema's en architectuurdocumenten van de oplossing. |
| [`Behuizingen/`](Behuizingen/) | Bestanden en ontwerpen voor de fysieke behuizing. |
| [`Code PI/`](Code%20PI/) | Hoofdcode voor de Raspberry Pi, inclusief GUI, LED-sturing en ultrasoon sensoren. |
| [`Datasets/`](Datasets/) | Informatie en links naar de datasets en model-artifacts op Kaggle. |
| [`Documentaties/`](Documentaties/) | Extra documentatie voor de Raspberry Pi en serveromgeving. |
| [`Pi_deploy_target96/`](Pi_deploy_target96/) | Deploy-bestanden voor het target96-model op de Raspberry Pi. |
| [`SocialeMedia/Poster/`](SocialeMedia/Poster/) | Poster- en presentatiemateriaal van het project. |
| [`RaspberryPI5_Pinout/`](RaspberryPI5_Pinout/) | Pinout-informatie voor de Raspberry Pi 5. |
| [`TestCodeC++/`](TestCodeC++/) | Testcode in C++ voor onderdelen van het systeem. |
| [`inference_gui.service`](inference_gui.service) | Systemd-service om de inferentie-GUI automatisch te starten. |
| [`start_garbage_gui.sh`](start_garbage_gui.sh) | Startscript voor de garbage detection GUI. |
| [`test_uart.py`](test_uart.py) | Python-testscript voor UART-communicatie. |

## Projectbeschrijving

Dit project implementeert een slimme afvalcontainer die:

- **Automatisch afvaltype detecteert** via camera en AI-objectdetectie.
- **Visueel aanduidt** welke container gebruikt moet worden met NeoPixel LED's.
- **Controleert of afval gevallen is** met ultrasoon sensoren.
- **Versnelde inferentie** ondersteunt via de Hailo AI Hat+.

## Hardware

| Component | Doel |
|---|---|
| Raspberry Pi 5 | Hoofd verwerkingseenheid |
| Hailo AI Hat+ | Hardware AI-acceleratie (optioneel) |
| Raspberry Pi Camera 3 | Objectdetectie |
| NeoPixel LED-strip | Visuele container-indicatie |
| Ultrasoon sensoren (HC-SR04) | Afvalniveaudetectie |

## GitHub en Kaggle verdeling

Deze repository bevat vooral:

- code
- documentatie
- configuratiebestanden
- het compacte referentie-modelpakket

Voor reproduceerbaarheid gebruiken we daarnaast Kaggle voor de grote bestanden:

- originele foto's: [Smart Bin Original Images](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-original-images)
- crops voor het finale model: [Smart Bin Classifier Crops](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-classifier-crops)
- zware modelbestanden en trainingsoutputs: [Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)

Met deze repo samen met die drie Kaggle-links kan het servergedeelte opnieuw worden opgebouwd.

## Gebruikte foto's en datastroom

De foto's zijn verzameld via:

- Google Dataset Search
- Kaggle
- [images.cv](https://images.cv/)

Een deel van de beelden kwam uit Amerikaanse datasets. Daardoor moesten we zelf foto's controleren en hersorteren, vooral voor `PMD`, zodat de sorteerlogica klopt met de regels in Brugge.

De flow voor het finale model is:

1. originele foto's
2. YOLO-detector zoekt het afvalobject
3. daaruit worden crops gemaakt
4. de two-stage classifier traint op die crops

## Serverdocumentatie

Voor de trainings- en exportflow:

- [Documentaties/Server/README.md](Documentaties/Server/README.md)
- [Documentaties/Server/Handleiding/README.md](Documentaties/Server/Handleiding/README.md)

## Links

- [Smart Bin Original Images](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-original-images)
- [Smart Bin Classifier Crops](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-classifier-crops)
- [Smart Bin Model Artifacts](https://www.kaggle.com/datasets/maartenaudenaert/smart-bin-model-artifacts)
- [Raspberry Pi Documentatie](https://www.raspberrypi.com/documentation/)
- [Pi Camera 3 Documentatie](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [Hailo RPi5 Voorbeelden](https://github.com/hailo-ai/hailo-rpi5-examples)
- [NeoPixel LED Guide](https://learn.adafruit.com/adafruit-neopixel-uberguide)

# 🗑️ SlimmeAfvalcontainer 

Een intelligente afvalcontainer met camera-gebaseerde objectdetectie, LED-indicatie en ultrasoon sensordetectie.

## 🎯 Projectbeschrijving

Dit project implementeert een slimme afvalcontainer die:

- **Automatisch afvaltype detecteert** via camera en AI-objectdetectie
- **Visueel aanduidt** welke container gebruikt moet worden met NeoPixel LED's
- **Controleert of afval gevallen is** met ultrasoon sensoren

## 💻 Hardware

### Componenten

- **Raspberry Pi 5** - Hoofd verwerkingseenheid
- **Raspberry Pi Camera 3** - Objectdetectie
- **NeoPixel LED-strip** - Visuele container-indicatie
- **Ultrasoon sensoren (HC-SR04)** - Afvalniveaudetectie
- **Diverse verbindingsmaterialen** - Bedrading, voeding, behuizing

## 🔧 Software

### Vereisten

- Python 3.10+
- Raspberry Pi OS
- Required Python libraries:
  - OpenCV (cv2)
  - TensorFlow / PyTorch
  - Adafruit NeoPixel
  - RPi.GPIO of gpiozero

### Datasets voor AI-training

We gebruiken de volgende datasets om het AI-model te trainen:

- **Custom Waste classification dataset** - [Kaggle](https://www.kaggle.com/datasets/wasifmahmood01/custom-waste-classification-dataset)
- **TrashNet** - [Kaggle](https://www.kaggle.com/datasets/feyzazkefe/trashnet/data)
- **Garbage Dataset** - [Kaggle](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

## 🚀 Functionaliteiten

### 1. Camera Objectdetectie

- Captureert beelden van de Pi Camera 3
- Herkent afvaltypes (papier, plastic, glas, etc.)
- Voert inferentie uit met getraind model

### 2. LED Indicatie (NeoPixel)

### 3. Ultrasoon Sensordetectie

- Meet afstand tot afval
- Detecteert wanneer afval in container valt
- Geeft aan wanneer de container vol zit

## 📁 Project Structuur
<!--
```
SlimmeAfvalcontainer/
├── README.md
├── requirements.txt
├── src/
│   ├── camera_detection.py      # Camera & AI-model integratie
│   ├── led_control.py            # NeoPixel LED controle
│   ├── ultrasoon_sensor.py       # Ultrasoon sensor interface
│   └── main.py                   # Hoofdprogramma
├── models/
│   └── waste_detector.tflite     # Getraind AI-model
├── config/
│   └── settings.yaml             # Configuratiebestand
└── tests/
    └── test_sensors.py           # Sensor-tests
```
-->
## 🔄 Workflow

```text
1. Camera legt beeld vast
   ↓
2. AI-model analyzeert afvaltype
   ↓
3. LED's geven visuele indicatie
   ↓
4. Ultrasoon sensor controleert afval-inlating
   ↓
5. System registreert event
```

## 🛠️ Gebruik
<!--
```bash
# Start het systeem
python src/main.py

# Start in debug mode
python src/main.py --debug
```
-->
## 📊 Configuratie
<!--
Bewerk `config/settings.yaml` voor:
- Camera resolutie
- LED kleurinstellingen
- Ultrasoon sensor sensitiviteit
- Model drempel (confidence)
-->
## 📝 Logging
<!--
Alle events worden gelogd in `logs/system.log`:
- Gedetecteerde afvaltypes
- LED-status veranderingen
- Sensorwaarden
- Fouten en waarschuwingen
-->

## 📈 Toekomstverbeteringen

- [ ] Cloud-connectiviteit voor monitoring
- [ ] Machine Learning model optimalisatie
- [ ] App voor gebruiker feedback
- [ ] Energie-optimalisatie
- [ ] Meerdere container-ondersteuning

## 👥 Team
- [<img src="https://github.com/MaartenAudenaert.png" alt="" width="25" style="margin-bottom:-6px;">Maarten Audenaert](https://github.com/MaartenAudenaert)
- [<img src="https://github.com/kobedemetser.png" alt="" width="25" style="margin-bottom:-6px;">Kobe Demetser](https://github.com/kobedemetser)
- [<img src="https://github.com/Oceandek.png" alt="" width="25" style="margin-bottom:-6px;">Ocean Dekeyser](https://github.com/Oceandek)
- [<img src="https://github.com/oppositemite.png" alt="" width="25" style="margin-bottom:-6px;">Juul Kerkhof](https://github.com/)
- [<img src="https://github.com/Bhavninderpalsingh-tech.png" alt="" width="25" style="margin-bottom:-6px;">Bhavninder Pal Singh](https://github.com/)



## 🔗 Links & Resources

- [Raspberry Pi Documentatie](https://www.raspberrypi.com/documentation/)
- [Pi Camera 3 Documentatie](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [NeoPixel LED Guide](https://learn.adafruit.com/adafruit-neopixel-uberguide)
- [Ultrasoon Sensor Gids](https://www.robotics-everywhere.com/hc-sr04-ultrasonic-sensor/)

---


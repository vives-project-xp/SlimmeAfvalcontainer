# Slimme Afvalcontainer

Een intelligente afvalcontainer met camera-gebaseerde objectdetectie, LED-indicatie en ultrasoon sensordetectie.

## Projectbeschrijving

Dit project implementeert een slimme afvalcontainer die:

- **Automatisch afvaltype detecteert** via camera en AI-objectdetectie (RF-DETR)
- **Visueel aanduidt** welke container gebruikt moet worden met NeoPixel LED's
- **Controleert of afval gevallen is** met ultrasoon sensoren
- **Versnelde inferentie** ondersteunt via de Hailo AI Hat+

## Hardware

| Component | Doel |
|---|---|
| Raspberry Pi 5 | Hoofd verwerkingseenheid |
| Hailo AI Hat+ | Hardware AI-acceleratie (optioneel) |
| Raspberry Pi Camera 3 | Objectdetectie |
| NeoPixel LED-strip | Visuele container-indicatie |
| Ultrasoon sensoren (HC-SR04) | Afvalniveaudetectie |

## Software – GUI starten

Het hoofdprogramma is `Code PI/garbagedetection_gui.py`.

```bash
# Standaard (model wordt automatisch gevonden)
python garbagedetection_gui.py

# Specifiek model opgeven
python garbagedetection_gui.py --model model_best_ema_target96.pth

# Volledig scherm (voor Pi touchscreen)
python garbagedetection_gui.py --fullscreen

# Camera rotatie + lagere drempel
python garbagedetection_gui.py --rotate 180 --threshold 0.4
```

**Toetsenbord snelkoppelingen**

| Toets | Actie |
|---|---|
| Spatie | Analyseer huidig camerabeeld |
| Escape / F11 | Toggle volledig scherm |

## AI-model prioriteit

De GUI laadt automatisch het beste beschikbare model in deze volgorde:

1. **Hailo HEF** (`.hef`) – snelste optie, vereist gecompileerd model voor Hailo AI Hat+
2. **RF-DETR** (`.pth`) – huidig productiemodel, draait op CPU
3. **Two-stage ONNX** – `stage1_main.onnx` + `stage2_overige.onnx`
4. **Single-stage ONNX** – `model.onnx` of `inference_model.onnx`

Modellen worden gezocht in: naast het script, `AI/`, en `../Ai-model/`.

## RF-DETR model

Het huidige model is getraind met RF-DETR (`model_best_ema_target96.pth`).  
Het modelbestand staat **niet** in git (`.gitignore`) vanwege de bestandsgrootte.

### Model kopiëren naar de Pi

```bash
scp model_best_ema_target96.pth <PI_USER>@<PI_IP>:~/SlimmeAfvalcontainer/Code\ PI/
```

### Testen zonder GUI (enkel inferentie op een afbeelding)

```bash
python infer_rfdetr_pi.py \
  --model model_best_ema_target96.pth \
  --image /pad/naar/test.jpg \
  --threshold 0.5 \
  --output prediction.jpg
```

## Hailo AI Hat+ (optioneel)

De Hailo AI Hat+ versnelt inferentie aanzienlijk. Hiervoor moet het model gecompileerd worden naar `.hef` formaat via de [Hailo Model Zoo / DFC toolchain](https://github.com/hailo-ai/hailo-rpi5-examples).

1. Exporteer RF-DETR naar ONNX
2. Compileer ONNX → HEF met de Hailo DFC
3. Zet het `.hef` bestand naast het script
4. De GUI pikt het automatisch op

## ONNX two-stage model (fallback)

Het twee-fase model staat op de Pi in:
```
/home/kobe/SlimmeAfvalcontainer/Code PI/AI/
```

Benodigde bestanden:
```
stage1_main.onnx
stage1_main.onnx.data
stage2_overige.onnx
stage2_overige.onnx.data
two_stage_metadata.json
```

**Inferentielogica:**
1. Stage 1 classificeert naar hoofdklasse (Organisch / PMD / Papier / Restafval / Overige)
2. Als resultaat `Overige` is → Stage 2 verfijnt naar subklasse (Batterijen, Glas, …)
3. Lage confidence → fallback naar `Restafval`

Labels worden gelezen uit `two_stage_metadata.json` – nooit hardgecodeerd.

## Installatie op de Pi

```bash
# Apt-pakketten (eenmalig)
sudo apt install python3-picamera2 python3-libcamera python3-pil.imagetk python3-tk

# Python-omgeving
cd ~/SlimmeAfvalcontainer/Code\ PI
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
pip install --upgrade pip
pip install rfdetr supervision pillow numpy onnxruntime
```

## Afvalklassen → containers

| Gedetecteerde klasse | Container |
|---|---|
| Organisch / GFT / Bio | Organisch (groen) |
| Papier / Karton | Papier (blauw) |
| PMD / Plastic / Metaal / Blik | PMD (geel) |
| Overige / Batterijen / … | Restafval (grijs) |
| Restafval | Restafval (grijs) |

## Datasets gebruikt voor training

- [Custom Waste Classification Dataset](https://www.kaggle.com/datasets/wasifmahmood01/custom-waste-classification-dataset)
- [TrashNet](https://www.kaggle.com/datasets/feyzazkefe/trashnet/data)
- [Garbage Classification v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

## Troubleshooting

**`python: command not found`** → gebruik `python3`

**Fout bij `pip install rfdetr`**
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements_pi.txt
```

**Inferentie te traag op Pi CPU**
- Verhoog threshold: `--threshold 0.6`
- Gebruik de Hailo AI Hat+ met een gecompileerd `.hef` model

**Camera start niet**
```bash
sudo apt install python3-picamera2 python3-libcamera
# Controleer of de camera ingeschakeld is in raspi-config
```

## Workflow

```
1. Camera legt beeld vast
        ↓
2. RF-DETR / Hailo detecteert afvalobject + bounding box
        ↓
3. Klasse wordt omgezet naar containertype
        ↓
4. LED-strip toont de juiste container
        ↓
5. Ultrasoon sensor bevestigt dat afval gevallen is
```

## Toekomst

- [ ] RF-DETR → HEF compileren voor Hailo AI Hat+ (snellere inferentie)
- [ ] Cloud-monitoring dashboard
- [ ] Mobiele app voor gebruikersfeedback
- [ ] Meerdere containers ondersteunen
- [ ] Energie-optimalisatie

## Team

- [Maarten Audenaert](https://github.com/MaartenAudenaert)
- [Kobe Demetser](https://github.com/kobedemetser)
- [Ocean Dekeyser](https://github.com/Oceandek)
- [Juul Kerkhof](https://github.com/)
- [Bhavninder Pal Singh](https://github.com/)

## Links

- [Raspberry Pi Documentatie](https://www.raspberrypi.com/documentation/)
- [Pi Camera 3 Documentatie](https://www.raspberrypi.com/documentation/accessories/camera.html)
- [Hailo RPi5 Voorbeelden](https://github.com/hailo-ai/hailo-rpi5-examples)
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [NeoPixel LED Guide](https://learn.adafruit.com/adafruit-neopixel-uberguide)

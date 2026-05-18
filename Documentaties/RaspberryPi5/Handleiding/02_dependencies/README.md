# 2. Dependencies installeren

De Pi heeft Python-dependencies nodig voor camera, GUI, ONNX-inference, LEDs en
GPIO.

## 2.1 Systeempackages

Installeer basispackages:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-tk git
```

Voor camera en GPIO zijn op Raspberry Pi OS meestal al veel packages aanwezig.
Controleer indien nodig:

```bash
sudo apt install -y python3-picamera2 python3-rpi.gpio
```

## 2.2 Virtual environment

In de huidige installatie wordt deze venv gebruikt:

```text
/home/kobe/SlimmeAfvalcontainer/venv_ssp
```

Aanmaken:

```bash
cd /home/kobe/SlimmeAfvalcontainer
python3 -m venv venv_ssp
source venv_ssp/bin/activate
```

## 2.3 Pythonpackages

Installeer minimaal:

```bash
pip install --upgrade pip
pip install numpy pillow opencv-python onnxruntime
pip install adafruit-circuitpython-neopixel rpi_ws281x RPi.GPIO
```

Als `opencv-python` problemen geeft op de Pi, gebruik dan de OS-package:

```bash
sudo apt install -y python3-opencv
```

## 2.4 Rechten voor LEDs

De NeoPixel-library heeft toegang tot GPIO/PWM nodig. Daarom draait de huidige
service als `root`.

Controleer de service:

```bash
systemctl cat garbagedetection-gui.service
```

Verwacht in de huidige setup:

```text
User=root
Group=root
```

## 2.5 ONNX Runtime controleren

Test in de venv:

```bash
source /home/kobe/SlimmeAfvalcontainer/venv_ssp/bin/activate
python -c "import onnxruntime as ort; print(ort.__version__)"
```

Als dit faalt, kan de GUI de modellen niet laden.


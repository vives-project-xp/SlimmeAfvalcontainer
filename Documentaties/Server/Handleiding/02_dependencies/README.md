# 2. Dependencies installeren

Dit onderdeel zet Python en alle packages klaar.

## 2.1 Python controleren

```bash
python3 --version
```

Gebruik bij voorkeur Python 3.10, 3.11 of 3.12. Als Python ontbreekt:

```bash
apt update
apt install -y python3 python3-venv python3-pip
```

## 2.2 Virtual environment maken

```bash
cd /root/smart_bin_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Vanaf nu moet je voor training telkens eerst activeren:

```bash
cd /root/smart_bin_project
source .venv/bin/activate
```

## 2.3 Packages installeren

Installeer de basisrequirements:

```bash
pip install -r requirements.txt
```

Bijhorend bestand:

- [requirements.txt](../../Bestanden/Scripts_en_config/requirements.txt)

Installeer aanvullend wat de trainingsscripts nodig hebben:

```bash
pip install ultralytics opencv-python pyyaml numpy
```

Korte uitleg:

- `torch` en `torchvision`: PyTorch packages om de classifier te trainen.
- `ultralytics`: package voor YOLO-training en export.
- `opencv-python`: wordt gebruikt om afbeeldingen te lezen en crops te maken.
- `onnx` en `onnxruntime`: nodig om modellen naar ONNX te exporteren en te testen.

## 2.4 PyTorch met CUDA

Controleer eerst:

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

Als dit `False` geeft terwijl je wel een NVIDIA GPU hebt, installeer dan een
CUDA-build van PyTorch die past bij je driver. Een veelgebruikte optie:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Controleer opnieuw:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'geen cuda')"
```

## 2.5 Ultralytics controleren

```bash
yolo checks
```

Of:

```bash
python3 -c "from ultralytics import YOLO; print('ultralytics ok')"
```

## 2.6 ONNX controleren

```bash
python3 -c "import onnx, onnxruntime; print('onnx ok')"
```

## 2.7 Schijfruimte controleren

```bash
df -h
```

Hou voldoende vrije ruimte over. Training kan mislukken als `/` of de projectdisk
vol raakt.

# 1. VM en projectmap maken

Dit onderdeel begint echt vanaf nul: je hebt nog geen VM of projectmap.

## 1.1 Kies een machine

Voor trainen heb je best een Linux VM of server met:

```text
Ubuntu 22.04 of 24.04
Python 3.10+
NVIDIA GPU met CUDA
minstens 16 GB RAM
voldoende opslagruimte
```

Aanbevolen opslag:

```text
minimaal 80 GB vrij
liever 150 GB of meer
```

Waarom zoveel? De dataset, YOLO-runs, cropdataset, checkpoints, ONNX exports en
logs nemen samen snel veel ruimte in.

## 1.2 GPU controleren

Op een nieuwe VM moet de NVIDIA driver werken:

```bash
nvidia-smi
```

Als dit faalt, moet eerst de NVIDIA driver/CUDA setup van de VM worden opgelost.
Zonder GPU kan veel nog wel draaien, maar training wordt traag.

## 1.3 SSH-toegang

Voorbeeld:

```bash
ssh root@<IP_OF_HOSTNAME>
```

In het huidige project was dat:

```bash
ssh root@100.95.124.116
```

Maar voor een nieuwe VM vervang je dit door het IP of de hostname van die nieuwe
machine.

## 1.4 Projectmap maken

Maak op de VM een projectmap:

```bash
mkdir -p /root/smart_bin_project
cd /root/smart_bin_project
```

Gebruik in deze handleiding overal:

```text
/root/smart_bin_project
```

Als je een andere map kiest, pas dan alle paden in scripts/configs aan.

## 1.5 Basisstructuur

Maak deze mappen:

```bash
mkdir -p data/Dataset
mkdir -p data/classifier_crops
mkdir -p models/two_stage_crops
mkdir -p runs
mkdir -p outputs
mkdir -p reports
mkdir -p logs
```

Verwachte basis:

```text
/root/smart_bin_project/
|-- data/
|-- models/
|-- runs/
|-- outputs/
|-- reports/
`-- logs/
```

## 1.6 Scripts in project zetten

Kopieer de scripts uit deze documentatie naar de projectroot:

- [train_two_stage_crops.py](../../Bestanden/Scripts_en_config/train_two_stage_crops.py)
- [export_two_stage_onnx.py](../../Bestanden/Scripts_en_config/export_two_stage_onnx.py)
- [train_two_stage.py](../../Bestanden/Scripts_en_config/train_two_stage.py)
- [train_yolo_l.py](../../Bestanden/Scripts_en_config/train_yolo_l.py)
- [train_yolo_until_target.py](../../Bestanden/Scripts_en_config/train_yolo_until_target.py)
- [verzamel_pi_bestanden.sh](../../Bestanden/Scripts_en_config/verzamel_pi_bestanden.sh)
- [requirements.txt](../../Bestanden/Scripts_en_config/requirements.txt)

Op een lokale machine kan dat bijvoorbeeld met `scp`:

```bash
scp Bestanden/Scripts_en_config/* root@<IP>:/root/smart_bin_project/
```

Of via GitHub: clone de repository en zorg dat de scripts in
`/root/smart_bin_project` terechtkomen.

## 1.7 Controle

Op de VM:

```bash
cd /root/smart_bin_project
ls -la
```

Je moet minstens zien:

```text
train_two_stage_crops.py
export_two_stage_onnx.py
requirements.txt
data/
models/
runs/
```


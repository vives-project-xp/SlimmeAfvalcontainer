# RF-DETR model naar Raspberry Pi zetten

Deze map bevat:
- `model_best_ema_target96.pth` (beste model)
- `infer_rfdetr_pi.py` (infer script op CPU)
- `requirements_pi.txt`

## 1) Vanaf deze server: map kopieren naar je Pi
Vervang `<PI_USER>` en `<PI_IP>`:

```bash
scp -r /root/rf-detr/pi_deploy_target96 <PI_USER>@<PI_IP>:~/
```

## 2) Op de Raspberry Pi: Python omgeving maken

```bash
cd ~/pi_deploy_target96
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_pi.txt
```

Opmerking:
- `rfdetr` installeert PyTorch dependency mee.
- Op een Pi is inferentie op CPU trager dan op GPU-server.

## 3) Test inferentie op 1 afbeelding

```bash
cd ~/pi_deploy_target96
source .venv/bin/activate
python infer_rfdetr_pi.py \
  --model model_best_ema_target96.pth \
  --image /pad/naar/test.jpg \
  --threshold 0.5 \
  --output prediction.jpg
```

## 4) Verwacht resultaat
- Terminal toont aantal detecties.
- Geannoteerde output staat in `prediction.jpg`.

## 5) Troubleshooting

### Fout: `python: command not found`
Gebruik `python3`.

### Fout bij `pip install rfdetr`
Controleer internet op de Pi en probeer:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements_pi.txt
```

### Te traag op Pi
- Gebruik hogere threshold (`--threshold 0.6`)
- Verklein input in je camera pipeline voor snellere inferentie
- Overweeg ONNX/TFLite optimalisatie als volgende stap

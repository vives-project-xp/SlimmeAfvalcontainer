# 6. Controle en troubleshooting

Gebruik dit hoofdstuk om snel te controleren of de Pi-runtime gezond is.

## 6.1 Service controleren

```bash
systemctl status garbagedetection-gui.service --no-pager -l
```

Goed:

```text
Active: active (running)
```

Fout:

```text
activating (auto-restart)
failed
```

Bekijk dan logs:

```bash
journalctl -u garbagedetection-gui.service -n 120 --no-pager
```

## 6.2 Proces controleren

```bash
ps -ef | grep -E 'garbagedetection_gui|start_garbage_gui' | grep -v grep
```

Verwacht iets zoals:

```text
/home/kobe/SlimmeAfvalcontainer/venv_ssp/bin/python /home/kobe/SlimmeAfvalcontainer/test11mei_light_crops/garbagedetection_gui.py --fullscreen
```

Door de symlink komt dit effectief uit bij:

```text
/home/kobe/SlimmeAfvalcontainer/finalmodel
```

## 6.3 Modelbestanden controleren

```bash
cd /home/kobe/SlimmeAfvalcontainer/finalmodel
ls -lah yolov8_detector.onnx stage1_main.onnx stage2_overige.onnx two_stage_metadata.json
```

Als een `.onnx.data` bestand ontbreekt, kan ONNX Runtime het model mogelijk niet
laden.

## 6.4 Python syntax controleren

```bash
cd /home/kobe/SlimmeAfvalcontainer/finalmodel
python3 -m py_compile garbagedetection_gui.py detector.py classifier.py led_controller.py ultrasone_controller.py
```

Geen output betekent normaal dat de syntax correct is.

## 6.5 Camera hangt bij LED-actie

Als de camera lijkt vast te hangen wanneer `Overige` gedetecteerd wordt, check
of `reject_blink` non-blocking is gemaakt.

In `led_controller.py` moet `_send_command_via_subprocess()` voor
`reject_blink` `subprocess.Popen(...)` gebruiken, niet alleen
`subprocess.run(...)`.

Controle:

```bash
grep -nE 'reject_blink|Popen|subprocess.run' /home/kobe/SlimmeAfvalcontainer/Code\ PI/led_controller.py
grep -nE 'reject_blink|Popen|subprocess.run' /home/kobe/SlimmeAfvalcontainer/finalmodel/led_controller.py
```

## 6.6 Foto's worden opgeslagen

De runtime hoort geen foto's meer op te slaan.

Controle:

```bash
find /home/kobe/SlimmeAfvalcontainer/finalmodel -maxdepth 1 -type d -name captures
grep -n "_save_capture_photo(pil_img)" /home/kobe/SlimmeAfvalcontainer/finalmodel/garbagedetection_gui.py
```

Als `captures` bestaat, kan die verwijderd worden:

```bash
sudo rm -rf /home/kobe/SlimmeAfvalcontainer/finalmodel/captures
```

## 6.7 Autostart na reboot testen

Herstart de Pi:

```bash
sudo reboot
```

Na opnieuw verbinden:

```bash
systemctl status garbagedetection-gui.service --no-pager -l
```

Als de service enabled is en de status running is, start de GUI correct mee op.

Controle enabled:

```bash
systemctl is-enabled garbagedetection-gui.service
```

Verwacht:

```text
enabled
```


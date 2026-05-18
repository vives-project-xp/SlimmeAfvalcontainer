# 4. GUI en autostart instellen

De GUI moet automatisch starten zodra de Pi grafisch opgestart is.

## 4.1 Hoofdservice

De huidige service heet:

```text
garbagedetection-gui.service
```

Controle:

```bash
systemctl status garbagedetection-gui.service --no-pager -l
```

Verwacht:

```text
Active: active (running)
```

## 4.2 Servicebestand

Locatie:

```text
/etc/systemd/system/garbagedetection-gui.service
```

Huidige kernconfiguratie:

```ini
[Unit]
Description=Slimme Afvalcontainer GUI
After=graphical.target
Wants=graphical.target

[Service]
Type=simple
User=root
Group=root
WorkingDirectory=/home/kobe/SlimmeAfvalcontainer
Environment=DISPLAY=:0
Environment=XAUTHORITY=/home/kobe/.Xauthority
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/kobe/SlimmeAfvalcontainer/start_garbage_gui.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical.target
```

## 4.3 Startscript

Locatie:

```text
/home/kobe/SlimmeAfvalcontainer/start_garbage_gui.sh
```

Het script start de GUI in fullscreen:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd /home/kobe/SlimmeAfvalcontainer/test11mei_light_crops
exec /home/kobe/SlimmeAfvalcontainer/venv_ssp/bin/python garbagedetection_gui.py --fullscreen
```

Als dit script nog naar `test11mei_light_crops` verwijst, moet die map bestaan
of als symlink naar `finalmodel` wijzen.

## 4.4 Service installeren of aanpassen

Na wijzigingen:

```bash
sudo systemctl daemon-reload
sudo systemctl enable garbagedetection-gui.service
sudo systemctl restart garbagedetection-gui.service
```

Controle:

```bash
systemctl status garbagedetection-gui.service --no-pager -l
```

## 4.5 Logs bekijken

Laatste logs:

```bash
journalctl -u garbagedetection-gui.service -n 80 --no-pager
```

Live volgen:

```bash
journalctl -u garbagedetection-gui.service -f
```

## 4.6 Veelvoorkomende fout

Als de service blijft herstarten met:

```text
cd: /home/kobe/SlimmeAfvalcontainer/test11mei_light_crops: No such file or directory
```

dan verwijst het startscript naar een oude map. Oplossing:

```bash
cd /home/kobe/SlimmeAfvalcontainer
ln -s finalmodel test11mei_light_crops
sudo systemctl restart garbagedetection-gui.service
```


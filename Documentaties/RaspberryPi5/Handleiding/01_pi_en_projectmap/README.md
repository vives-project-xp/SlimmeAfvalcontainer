# 1. Pi en projectmap klaarzetten

Dit onderdeel begint vanaf een Raspberry Pi 5 waarop Raspberry Pi OS al
geinstalleerd is.

## 1.1 Benodigdheden

Aanbevolen hardware:

```text
Raspberry Pi 5
Raspberry Pi Camera Module
Touchscreen of HDMI-scherm
4 WS2812B LED-strips
4 HC-SR04 ultrasone sensoren
5V voeding voor de LED-strips
netwerktoegang via Wi-Fi of ethernet
```

## 1.2 SSH-toegang

Voor de huidige Pi:

```bash
ssh kobe@10.10.226.229
```

Wachtwoord in de huidige testopstelling:

```text
kobe
```

Gebruik bij een andere Pi het juiste IP-adres.

## 1.3 Projectmap

De huidige projectmap op de Pi is:

```text
/home/kobe/SlimmeAfvalcontainer
```

Controle:

```bash
ls -la /home/kobe/SlimmeAfvalcontainer
```

Je moet onder andere deze onderdelen verwachten:

```text
Code PI/
finalmodel/
start_garbage_gui.sh
```

## 1.4 Modelmap en compatibele symlink

De huidige runtime gebruikt:

```text
/home/kobe/SlimmeAfvalcontainer/finalmodel
```

Omdat oude startbestanden vroeger naar `test11mei_light_crops` verwezen, bestaat
er een symlink:

```bash
cd /home/kobe/SlimmeAfvalcontainer
ln -s finalmodel test11mei_light_crops
```

Controle:

```bash
ls -ld /home/kobe/SlimmeAfvalcontainer/test11mei_light_crops
```

Verwachte output:

```text
test11mei_light_crops -> finalmodel
```

Zonder deze symlink kan de service falen als het startshellscript nog het oude
pad gebruikt.

## 1.5 Camera controleren

Controleer of de camera zichtbaar is:

```bash
rpicam-hello --list-cameras
```

Of test kort:

```bash
rpicam-hello -t 3000
```

Als dit niet werkt, moet eerst de camera-configuratie opgelost worden.


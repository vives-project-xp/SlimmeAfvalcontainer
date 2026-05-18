# 5. LEDs en ultrasoon controleren

De Pi gebruikt LED-strips om de gekozen bak te tonen en ultrasone sensoren om
vulgraad en inworp te detecteren.

## 5.1 LED-pinnen

GPIO-mapping volgens de huidige `led_controller.py`:

| Bak | GPIO | Standaardkleur |
|---|---:|---|
| Restafval | GPIO18 | rood |
| Karton/Papier | GPIO13 | blauw |
| Organisch | GPIO12 | groen |
| PMD | GPIO19 | geel |

Detaildocumentatie:

- [hardware/README.md](hardware/README.md)
- [hardware/ledstrip.md](hardware/ledstrip.md)
- [hardware/bedrading.md](hardware/bedrading.md)

## 5.2 LED-gedrag bij classificatie

De GUI vertaalt labels naar een bak of reject-status.

| Modelresultaat | LED-gedrag |
|---|---|
| `Organisch` | Organisch groen, andere bakken rood |
| `PMD` | PMD groen, andere bakken rood |
| `Papier` | Karton/Papier groen, andere bakken rood |
| `Restafval` | Restafval groen, andere bakken rood |
| `Overige` | alle LEDs rood knipperen en rood blijven |
| `Overige/<subklasse>` | alle LEDs rood knipperen en rood blijven |

De subklassen gaan dus bewust naar geen enkele bak:

```text
Overige/Batterijen
Overige/Elektronica
Overige/Glas
Overige/Lightbulbs
Overige/Metaal
```

## 5.3 Waarom rood bij Overige?

`Overige` en alle subklassen betekenen dat de afvalcontainer geen veilige
standaardbak mag kiezen. De LEDs geven daarom een reject-signaal:

```text
reject_blink
```

Dat commando start het knipperen in een subproces, zodat de camera en GUI niet
blijven wachten tot de LED-animatie klaar is.

## 5.4 Oranje flash

Oranje betekent:

```text
inworp bevestigd
```

De ultrasone sensor heeft dan gedetecteerd dat er effectief iets in de gekozen
bak gegooid werd.

Bij `Overige` hoort er normaal geen bak gekozen te worden en dus ook geen
inworp-confirmatie voor een specifieke bak.

## 5.5 Ultrasone pinnen

GPIO-mapping volgens de huidige documentatie:

| Bak | TRIG | ECHO |
|---|---:|---:|
| Restafval | GPIO25 | GPIO26 |
| PMD | GPIO22 | GPIO4 |
| Papier | GPIO23 | GPIO24 |
| Organisch | GPIO17 | GPIO27 |

Detaildocumentatie:

- [hardware/README.md](hardware/README.md)
- [hardware/ultrasone.md](hardware/ultrasone.md)
- [hardware/bedrading.md](hardware/bedrading.md)

## 5.6 Snelle LED-test

Op de Pi:

```bash
cd /home/kobe/SlimmeAfvalcontainer/finalmodel
sudo LED_PROCESS_ISOLATION=0 /home/kobe/SlimmeAfvalcontainer/venv_ssp/bin/python -c "from led_controller import LedController; c=LedController(); print(c.send_command('select_pmd'))"
```

Alles uit:

```bash
sudo LED_PROCESS_ISOLATION=0 /home/kobe/SlimmeAfvalcontainer/venv_ssp/bin/python -c "from led_controller import LedController; c=LedController(); print(c.send_command('off'))"
```

Reject-test:

```bash
sudo LED_PROCESS_ISOLATION=0 /home/kobe/SlimmeAfvalcontainer/venv_ssp/bin/python -c "from led_controller import LedController; c=LedController(); print(c.send_command('reject_blink'))"
```

# LED-strip documentatie

## Overzicht

`led_controller.py` stuurt 4 WS2812B LED-strips aan via de GPIO-pinnen van de Raspberry Pi. Elke strip stelt één afvalcategorie voor en krijgt een eigen kleur. De controller biedt een eenvoudige `send_command(cmd)` API die door andere modules (bv. `inference_gui.py`) gebruikt wordt.

---

## Hardware

### Voeding

De strips worden gevoed door een **Meanwell RS-25-5** (5V, 5A). Per strip is een **1000 µF elektrolytische condensator** geplaatst tussen de voeding en de strip om inrushstroom bij het opstarten op te vangen en spanningspieken te dempen. Er zijn dus **4 condensators** in totaal.

**Aansluitschema per strip:**

```
Meanwell RS-25-5
  +5V ──┬─────────────────────────────── +5V ingang LED-strip
       (+)
    condensator
      1000µF
       (−)
        │
  GND ──┴─────────────────────────────── GND ingang LED-strip
```

De condensator hangt **parallel** over de voedingsrail: (+) naar +5V, (−) naar GND. Hij staat dus niet in serie in de stroomweg, maar filtert spanningspieken op de rail.

> Let op de polariteit: de positieve pool naar +5V, de negatieve pool naar GND.

### DIN-lijn weerstand

Op elke **DIN-lijn** (datasignaal van de Raspberry Pi naar de strip) zit een **220 Ω weerstand** in serie. Dit beschermt de eerste LED-driver tegen te hoge instroompieken en vermindert reflecties op de signaalleiding.

```
GPIO-pin ──[ 220 Ω ]──── DIN LED-strip
```

### GPIO pin-mapping (BCM)

| Strip naam   | GPIO pin | Aantal LEDs |
|--------------|----------|-------------|
| `rest`       | GPIO 18  | 51          |
| `karton`     | GPIO 13  | 38          |
| `organisch`  | GPIO 12  | 51          |
| `pmd`        | GPIO 19  | 51          |

> De GPIO-nummering volgt het BCM-schema (niet de fysieke pin-nummers).

### Kleurcodering per afvalcategorie

| Categorie    | Kleur         | RGB-waarde      |
|--------------|---------------|-----------------|
| `rest`       | Rood          | (255, 70, 70)   |
| `karton`     | Blauw         | (70, 140, 255)  |
| `organisch`  | Groen         | (0, 255, 0)     |
| `pmd`        | Geel          | (255, 190, 40)  |

### Speciale kleuren

| Toestand          | Kleur              | RGB-waarde       |
|-------------------|--------------------|------------------|
| Correct (groen)   | Groen              | (0, 255, 0)      |
| Fout (rood)       | Rood               | (255, 0, 0)      |
| Uit               | Uit                | (0, 0, 0)        |

### Overige afvaltypes (speciale kleurenmap)

| Commando                  | Kleur              | RGB-waarde        |
|---------------------------|--------------------|-------------------|
| `overige`                 | Lichtgrijs         | (160, 160, 160)   |
| `overige_batterijen`      | Oranje             | (255, 120, 40)    |
| `overige_elektronica`     | Rood-oranje        | (255, 60, 60)     |
| `overige_glas`            | Teal               | (40, 190, 170)    |
| `overige_lightbulbs`      | Geel               | (255, 230, 90)    |
| `overige_metaal`          | Bruin              | (150, 120, 95)    |

---

## Installatie & vereisten

```bash
pip install adafruit-circuitpython-neopixel rpi_ws281x
```

De neopixel-library vereist toegang tot de GPIO-hardware. Draai het script als root of via `sudo` als GPIO-permissies dit vereisen.

---

## Configuratie via omgevingsvariabelen

Alle instellingen hebben een standaardwaarde maar kunnen overschreven worden via omgevingsvariabelen:

| Variabele               | Standaard | Beschrijving                                      |
|-------------------------|-----------|---------------------------------------------------|
| `LED_BRIGHTNESS`        | `0.25`    | Helderheid van alle strips (0.0 – 1.0)            |
| `LED_MAX_CURRENT_A`     | `5.0`     | Maximale stroomtoevoer in ampère                  |
| `LED_CURRENT_HEADROOM`  | `0.85`    | Veiligheidsmarge op de max stroom (bijv. 85%)     |
| `LED_COUNT_REST`        | `51`      | Aantal LEDs op de REST-strip                      |
| `LED_COUNT_KARTON`      | `38`      | Aantal LEDs op de KARTON-strip                    |
| `LED_COUNT_ORGANISCH`   | `51`      | Aantal LEDs op de ORGANISCH-strip                 |
| `LED_COUNT_PMD`         | `51`      | Aantal LEDs op de PMD-strip                       |
| `LED_PROCESS_ISOLATION` | `1`       | Subproces-isolatie aan (`1`) of uit (`0`)         |

Voorbeeld:
```bash
LED_BRIGHTNESS=0.5 LED_MAX_CURRENT_A=3.0 python led_controller.py
```

---

## Stroombeveiliging (current limiting)

De controller berekent voor elke strip de geschatte stroomopname op basis van de RGB-waarden. Als de totale geschatte stroom de limiet overschrijdt, worden de RGB-waarden proportioneel teruggeschaald.

**Formule:**
```
I_per_led = (R + G + B) / 255 * 20mA
I_totaal  = I_per_led * aantal_leds
```

Als `I_totaal > MAX_CURRENT * HEADROOM`, worden R, G en B vermenigvuldigd met de schaalfactor:
```
schaal = limiet / I_totaal
```

De controller logt dit in de terminal:
```
[LED] Current limit toegepast voor rest: (255, 70, 70) -> (180, 49, 49) (est 6.23A > 4.25A)
```

---

## Subproces-isolatie (process isolation)

Standaard (`LED_PROCESS_ISOLATION=1`) stuurt de controller elk commando door naar een **apart Python-subproces**. Dit beschermt de GUI tegen crashes die veroorzaakt worden door de native neopixel/GPIO-bibliotheek.

- Het kindproces initialiseert zijn eigen `LedController` met `LED_PROCESS_ISOLATION=0`
- Het resultaat wordt via `stdout` teruggegeven aan het hoofdproces
- Timeout per commando: 5 seconden

Zet isolatie uit voor hogere snelheid of bij directe hardware-integratie:
```bash
LED_PROCESS_ISOLATION=0 python led_controller.py
```

---

## API-referentie

### `LedController(led_count_map=None, brightness=0.25)`

Initialiseer de controller. Bij een mislukte GPIO-init blijft de controller actief maar worden LED-commando's genegeerd (de GUI crasht niet).

### `send_command(cmd: str) -> str`

Stuur een commando naar de strips. Geeft een statusstring terug.

#### Ondersteunde commando's

| Commando              | Gedrag                                                                                  |
|-----------------------|-----------------------------------------------------------------------------------------|
| `pmd`                 | Enkel de PMD-strip aan in geel, rest uit                                                |
| `rest`                | Enkel de REST-strip aan in rood, rest uit                                               |
| `karton` / `papier`   | Enkel de KARTON-strip aan in blauw, rest uit (`papier` is alias voor `karton`)          |
| `organisch` / `bio`   | Enkel de ORGANISCH-strip aan in groen, rest uit (`bio` is alias voor `organisch`)       |
| `all`                 | Alle 4 strips aan in hun eigen kleur                                                    |
| `idle`                | Zelfde als `all` — alle strips aan in hun eigen kleur                                   |
| `select_<naam>`       | Geselecteerde strip groen, de overige drie rood (visuele bevestiging juiste bak)        |
| `reject`              | Alle strips rood (ongekend of fout afval)                                               |
| `hit`                 | Herhaal de laatst gekozen categorie (nuttig bij opeenvolgende detecties)                |
| `off`                 | Alle strips uit                                                                         |
| `reset`               | Alle strips uit + reset huidige keuze naar `NONE`                                       |
| `overige`             | Alle strips in lichtgrijs                                                               |
| `overige_batterijen`  | Alle strips in oranje                                                                   |
| `overige_elektronica` | Alle strips in rood-oranje                                                              |
| `overige_glas`        | Alle strips in teal                                                                     |
| `overige_lightbulbs`  | Alle strips in geel                                                                     |
| `overige_metaal`      | Alle strips in bruin                                                                    |

#### Returnwaarden

| Return            | Betekenis                          |
|-------------------|------------------------------------|
| `OK: <CMD>`       | Commando succesvol uitgevoerd      |
| `UNKNOWN: <cmd>`  | Onbekend commando                  |
| `ERROR: <cmd>`    | Fout tijdens uitvoering            |

### `close()`

Zet alle strips uit en geeft GPIO-resources vrij. Altijd aanroepen bij afsluiten.

---

## Gebruik als module

```python
from led_controller import LedController

ctrl = LedController()

ctrl.send_command("pmd")        # PMD-strip aan
ctrl.send_command("select_pmd") # PMD groen, rest rood
ctrl.send_command("all")        # Alle strips aan
ctrl.send_command("off")        # Alles uit

ctrl.close()
```

---

## Standalone testen

```bash
python "Code PI/led_controller.py"
```

Dit doorloopt automatisch de commando's `pmd → rest → karton → organisch → hit → off` met 2 seconden tussentijd.

---

## Klasse-diagram

```
LedController
├── __init__(led_count_map, brightness)
│   ├── proxy_mode: LED-IO via subprocess
│   └── direct: neopixel strips initialiseren
│
├── send_command(cmd) -> str
│   ├── _send_command_via_subprocess(cmd)  [proxy mode]
│   └── directe strip-aanturing            [direct mode]
│
├── Interne methodes
│   ├── _show_bin(bin_name)         – één strip aan, rest uit
│   ├── _show_all_bins()            – alle strips in eigen kleur
│   ├── _show_all_color(rgb)        – alle strips zelfde kleur
│   ├── _show_selection(bin_name)   – groen/rood feedback
│   ├── _set_all(rgb)               – alle strips zelfde kleur (laag niveau)
│   ├── _limit_color_for_strip()    – current limiting
│   └── _estimate_led_current_a()   – stroomschatting
│
└── close()
```

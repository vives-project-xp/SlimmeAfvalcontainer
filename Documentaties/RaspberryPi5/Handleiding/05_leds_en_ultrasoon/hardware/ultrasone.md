# Ultrasone sensor documentatie

## Overzicht

`ultrasone_controller.py` bewaakt de vullingsgraad van 4 afvalbakken via HC-SR04 ultrasonische sensoren. De metingen lopen continu in een **achtergrond-thread**. Andere modules (bv. `inference_gui.py`) halen de meest recente status op via de `snapshot()` methode.

---

## Hardware

### Sensoren

Elke bak heeft één HC-SR04 sensor met een **TRIG**-pin (trigger) en een **ECHO**-pin. De sensor is diagonaal in de bakhoek geplaatst; de kalibratie-afstand voor een lege bak is daarom langer dan de verticale bakhoogte.

### GPIO pin-mapping (BCM)

| Bak       | TRIG pin | ECHO pin |
|-----------|----------|----------|
| `rest`    | GPIO 25  | GPIO 26  |
| `pmd`     | GPIO 22  | GPIO 4   |
| `papier`  | GPIO 23  | GPIO 24  |
| `org`     | GPIO 17  | GPIO 27  |

> De GPIO-nummering volgt het BCM-schema (niet de fysieke pin-nummers).

### Aansluitschema per sensor

De HC-SR04 werkt op 5V en geeft een 5V ECHO-signaal terug. De Raspberry Pi GPIO-pinnen zijn echter **3,3V-logica**. Om de GPIO te beschermen wordt een **spanningsdeler** gebruikt op de ECHO-lijn.

**Spanningsdeler (ECHO):**

```
HC-SR04 ECHO (5V)
        │
      [1kΩ]
        │
        ├──── GPIO-pin (Raspberry Pi)
        │
      [2kΩ]
        │
       GND
```

Uitgangsspanning: `5V × 2kΩ / (1kΩ + 2kΩ) ≈ 3,33V` — veilig voor de Raspberry Pi GPIO.

De TRIG-lijn heeft **geen** spanningsdeler nodig: de Pi stuurt 3,3V uit en de HC-SR04 accepteert dit als geldig HIGH-signaal.

**Volledig schema per sensor:**

```
Raspberry Pi GPIO          HC-SR04
  TRIG-pin ─────────────── TRIG
  ECHO-pin ──[1kΩ]──┬───── ECHO
                  [2kΩ]
                    │
                   GND
  GND      ─────────────── GND
  5V       ─────────────── VCC
```

---

## Installatie & vereisten

```bash
pip install RPi.GPIO
```

De module vereist toegang tot GPIO-hardware. Draai het script als root of via `sudo` als GPIO-permissies dit vereisen.

---

## Configuratie

De drempelwaarden zijn klasseconstanten in `UltrasonicMonitor`:

| Constante               | Standaard | Beschrijving                                                        |
|-------------------------|-----------|---------------------------------------------------------------------|
| `EMPTY_DISTANCE_CM`     | `75.0`    | Gekalibreerde afstand (cm) bij een volledig lege bak                |
| `THROW_THRESHOLD_CM`    | `10.0`    | Minimale afstandsdaling (cm) om een inworp te registreren           |
| `THROW_DEBOUNCE_S`      | `1.5`     | Minimale tijd (s) tussen twee opeenvolgende inworp-detecties        |
| `THROW_CONFIRM_DELAY_S` | `0.8`     | Wachttijd (s) vóór bevestigingsmeting na een mogelijke inworp       |
| `THROW_CONFIRM_SAMPLES` | `5`       | Aantal samples voor de bevestigingsmeting                           |
| `THROW_STATUS_HOLD_S`   | `2.0`     | Hoe lang (s) de status "inworp gedetecteerd" zichtbaar blijft       |

---

## Vullingsgraad berekening

De fill-percentage wordt berekend op basis van de gemeten afstand ten opzichte van de lege-bak-referentie:

**Formule:**
```
fill_pct = ((EMPTY_DISTANCE_CM - gemeten_cm) / EMPTY_DISTANCE_CM) * 100
```

Geldige meetafstand: tussen **2 cm** en **400 cm**. Buiten dit bereik geldt de meting als ongeldig (`None`).

Een bak wordt als **vol** beschouwd als:
- gemeten afstand < 5 cm, **of**
- `fill_pct >= 95`

---

## Inworp-detectie

De controller detecteert een inworp via een twee-staps confirmatieprocedure om valse positieven te vermijden:

1. **Drempelcheck**: de afstand daalt meer dan `THROW_THRESHOLD_CM` ten opzichte van de vorige meting.
2. **Debounce**: er zijn minstens `THROW_DEBOUNCE_S` seconden verstreken sinds de laatste inworp.
3. **Bevestiging**: na `THROW_CONFIRM_DELAY_S` seconden wordt opnieuw gemeten (`THROW_CONFIRM_SAMPLES` samples). De daling moet nog steeds aanwezig zijn.
4. **Resultaat**: bij bevestiging wordt `threw_object = True` en wordt de tijdstempel bijgewerkt.

---

## API-referentie

### `UltrasonicMonitor(debug=False, debug_every_cycles=1, sensor_overrides=None)`

Start de achtergrond-thread en initialiseert de GPIO-pinnen.

| Parameter            | Type                                        | Beschrijving                                                               |
|----------------------|---------------------------------------------|----------------------------------------------------------------------------|
| `debug`              | `bool`                                      | Zet debug-uitvoer aan in de terminal                                       |
| `debug_every_cycles` | `int`                                       | Druk debug-info elke N meetcycli af (standaard elke cyclus)                |
| `sensor_overrides`   | `dict[str, dict[str, int] \| None] \| None` | Overschrijf de TRIG/ECHO-pinnen per bak; geef `None` om een bak uit te schakelen |

Bij een mislukte GPIO-initialisatie blijft het object actief maar worden metingen overgeslagen (`enabled = False`).

**Voorbeeld met pin-override:**
```python
monitor = UltrasonicMonitor(
    sensor_overrides={"org": {"trig": 6, "echo": 5}}
)
```

**Voorbeeld met bak uitschakelen:**
```python
monitor = UltrasonicMonitor(
    sensor_overrides={"org": None}
)
```

---

### `snapshot() -> dict[str, dict[str, object]]`

Geeft een momentopname van de huidige sensorstatus terug. Thread-safe.

**Returnstructuur:**
```python
{
    "rest": {
        "fill_pct": int | None,  # Vulgraad 0–100, of None bij geen meting
        "is_full":  bool,        # True als bak vol is
        "threw_object": bool,    # True bij vers gedetecteerde inworp
        "text": str,             # Leesbare statustekst
    },
    # ook "pmd", "papier", "org"
}
```

**Mogelijke statusteksten:**

| Toestand                                | `text`                          |
|-----------------------------------------|---------------------------------|
| Geen geldige meting                     | `"geen meting"`                 |
| Bak vol                                 | `"VOL (97%)"`                   |
| Inworp zojuist gedetecteerd             | `"inworp gedetecteerd (42%)"`   |
| Bak bijna vol (≥ 80 %)                  | `"bijna vol (83%)"`             |
| Normale toestand                        | `"45%"`                         |
| Sensor uitgeschakeld of niet aanwezig   | `"n.v.t."`                      |
| Status nog onbekend (bij opstart)       | `"onbekend"`                    |

---

### `close()`

Stopt de achtergrond-thread en maakt de GPIO-resources vrij via `GPIO.cleanup()`. Altijd aanroepen bij afsluiten.

---

## Debug-modus

Zet `debug=True` om per meetcyclus een samenvatting in de terminal te printen:

```
[Ultrasoon DEBUG] rest   trig=25 echo=26 distance=  38.4cm fill= 48 throw=False status=48%
[Ultrasoon DEBUG] pmd    trig=22 echo= 4 distance=  12.1cm fill= 83 throw=False status=bijna vol (83%)
[Ultrasoon DEBUG] papier trig=23 echo=24 distance=   4.2cm fill=100 throw=False status=VOL (100%)
[Ultrasoon DEBUG] org    trig=17 echo=27 distance=  None   fill= -1 throw=False status=geen meting
[Ultrasoon DEBUG] ------------------------------------------------------------------------
```

Met `debug_every_cycles=10` wordt de uitvoer slechts elke 10 cycli geprint.

---

## Gebruik als module

```python
from ultrasone_controller import UltrasonicMonitor

monitor = UltrasonicMonitor()

status = monitor.snapshot()
print(status["rest"]["fill_pct"])      # bv. 48
print(status["pmd"]["threw_object"])   # bv. True
print(status["papier"]["text"])        # bv. "bijna vol (83%)"

monitor.close()
```

---

## Klasse-diagram

```
UltrasonicMonitor
├── __init__(debug, debug_every_cycles, sensor_overrides)
│   ├── GPIO-pinnen instellen (TRIG/ECHO per bak)
│   └── achtergrond-thread starten
│
├── snapshot() -> dict
│   └── thread-safe kopie van _status teruggeven
│
├── Interne methodes
│   ├── _loop()                         – meetlus in achtergrond-thread
│   ├── _measure_distance(cfg)          – enkele HC-SR04 meting (cm of None)
│   ├── _measure_average(cfg, samples)  – gemiddelde van N metingen
│   ├── _calc_fill_pct(distance_cm)     – omrekenen naar vulpercentage
│   ├── _detect_throw_event(...)        – inworp-detectie met bevestiging
│   └── _debug_print(msg)               – conditionele terminal-uitvoer
│
└── close()
```

# Bedrading en softwareconfig WS2812B (4 strips direct op Raspberry Pi)

Deze setup gebruikt geen ESP32 en geen UART meer.
De GUI stuurt de leds direct aan via [Code PI/led_controller.py](Code%20PI/led_controller.py).

## GPIO pinnen voor 4 aparte strips

- REST -> GPIO18 (fysieke pin 12)
- KARTON -> GPIO13 (fysieke pin 33)
- ORGANISCH -> GPIO12 (fysieke pin 32)
- PMD -> GPIO19 (fysieke pin 35)

Deze mapping komt exact overeen met de softwareconfig.

## Aansluiten per strip

Per strip heb je 3 verbindingen:

- 5V voeding -> 5V van de strip
- GND -> GND van de strip
- DATA van gekozen GPIO -> DIN van de strip

## Verplichte extra componenten voor WS2812B

- Plaats een serieweerstand van 330-470 ohm in de datalijn van elke strip.
- Plaats een elco van 1000 uF (minimaal 6.3V) tussen 5V en GND bij de start van de strips.
- Gebruik een externe 5V voeding voor de strips (niet alleen via Pi 5V pin bij veel leds).
- Verbind alle gronden met elkaar: Pi GND + voeding GND + alle strip GND.

## Spanningsniveau data (belangrijk)

De Raspberry Pi gebruikt 3.3V datalogica.
WS2812B op 5V werkt soms direct, maar voor stabiele werking is een level shifter sterk aanbevolen:

- 74AHCT125 of 74HCT14 (3.3V -> 5V) op elke datalijn.

## Voeding snelcheck

Rekenregel: maximaal ongeveer 60 mA per led bij vol wit op 100%.

Formule:

- I_max = aantal_leds * 0.06 A

Voorbeeld bij 191 leds totaal:

- 191 * 0.06 = 11.46 A (theoretisch maximum)

In de praktijk vaak lager door brightness-limiet, maar kies voeding met marge.

## Software vereisten (Pi)

Benodigde Python packages in je venv:

- adafruit-circuitpython-neopixel
- rpi_ws281x

En op OS-niveau:

- Python met GPIO toegang
- start de app bij voorkeur met sudo voor stabiele ws281x toegang

## Runtime configuratie

De led-controller ondersteunt deze omgevingsvariabelen:

- LED_COUNT_REST (standaard: 51)
- LED_COUNT_KARTON (standaard: 38)
- LED_COUNT_ORGANISCH (standaard: 51)
- LED_COUNT_PMD (standaard: 51)
- LED_BRIGHTNESS (standaard: 0.25)
- LED_MAX_CURRENT_A (standaard: 5.0)
- LED_CURRENT_HEADROOM (standaard: 0.85)

Voorbeeld:

- LED_COUNT_REST=51 LED_COUNT_KARTON=38 LED_COUNT_ORGANISCH=51 LED_COUNT_PMD=51 LED_BRIGHTNESS=0.20 LED_MAX_CURRENT_A=5.0 LED_CURRENT_HEADROOM=0.85 sudo -E /pad/naar/python [Code PI/inference_gui.py](Code%20PI/inference_gui.py)

Stroomlimiter in software:

- De controller schat stroom op basis van RGB-waarde en aantal leds van de actieve strip.
- Als de geschatte stroom boven de limiet komt, schaalt de software de RGB-waarde automatisch omlaag.
- Effectieve limiet = LED_MAX_CURRENT_A * LED_CURRENT_HEADROOM.

Huidige fysieke setup:

- REST: 51 leds
- KARTON: 38 leds
- ORGANISCH: 51 leds
- PMD: 51 leds

## Commando mapping vanuit GUI

De GUI stuurt commando's naar de controller:

- organisch -> ORGANISCH strip aan
- pmd -> PMD strip aan
- karton -> KARTON strip aan
- rest -> REST strip aan
- papier -> alias van karton
- bio -> alias van organisch
- off -> alle strips uit
- reset -> alle strips uit en keuze reset

Bij een nieuw commando gaat eerst alles uit, daarna alleen de gekozen strip aan.

## HC-SR04 ultrasonische sensors (4x)

Onderstaande pinout komt overeen met de originele `Ultrasone_controller.py`:

- Sensor 1 (Restafval): TRIG -> GPIO25 (pin 22), ECHO -> GPIO26 (pin 37)
- Sensor 2 (PMD): TRIG -> GPIO22 (pin 15), ECHO -> GPIO4 (pin 7)
- Sensor 3 (Papier): TRIG -> GPIO23 (pin 16), ECHO -> GPIO24 (pin 18)
- Sensor 4 (Glas): TRIG -> GPIO17 (pin 11), ECHO -> GPIO27 (pin 13)

Power voor alle HC-SR04:
- VCC -> 5V (pin 2 of 4)
- GND -> GND (bijv. pin 6, 9, 14, ...)

Belangrijk: ECHO is 5V output. Gebruik altijd een niveauadapter of een weerstandendeler naar 3.3V voor elke ECHO-lijn voordat je het op Pi GPIO input aansluit.

TRIG kan direct vanaf Pi 3.3V output (HC-SR04 erkent dit als hoog genoeg).

### Ultrasonische timing in Python

1. Zet TRIG output laag (2 ms)
2. Zet TRIG ~10 µs hoog
3. Zet TRIG laag
4. Meet hoelang ECHO hoog blijft
5. Afstand (cm) = (duration_microsec / 58)

### Cross-talk voorkomen

- Trigger sensoren één voor één (niet tegelijk), met kleine pauze tussen de metingen.
- Zorg dat je aansturing in software sequentieel gebeurt (sensor 1 -> 2 -> 3 -> 4) en niet parallel, anders kun je echo's verwisselen.

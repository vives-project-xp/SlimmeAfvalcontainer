# Bedrading WS2812B (4 strips direct op Raspberry Pi)

Deze setup gebruikt geen ESP32 en geen UART meer. De 4 WS2812B-strips zitten direct op de Raspberry Pi.

## GPIO pinnen voor 4 aparte strips

- REST -> GPIO18 (pin 12)
- KARTON -> GPIO13 (pin 33)
- ORGANISCH -> GPIO12 (pin 32)
- PMD -> GPIO19 (pin 35)

Dit zijn stabiele keuzes op de Pi voor LED-data met gangbare WS281x-drivers.

## Aansluiten per strip

Per strip heb je 3 verbindingen:

- 5V voeding -> 5V van de strip
- GND -> GND van de strip
- DATA van gekozen GPIO -> DIN van de strip

## Verplichte extra componenten voor WS2812B

- Plaats een serieweerstand van 330-470 ohm in de datalijn van elke strip.
- Plaats een elco van 1000 uF (minimaal 6.3V) tussen 5V en GND bij het begin van de strips.
- Gebruik een externe 5V voeding voor de strips (niet alleen via Pi 5V pin voor grote aantallen leds).
- Verbind alle gronden met elkaar: Pi GND + voeding GND + alle strip GND.

## Spanningsniveau data (belangrijk)

De Raspberry Pi geeft 3.3V datalogica. WS2812B op 5V kan dit soms accepteren, maar niet altijd stabiel.

Voor betrouwbare werking:

- Gebruik een 74AHCT125 of 74HCT14 level shifter (3.3V -> 5V) voor elke datalijn.

## Voeding snelcheck

Rekenregel: maximaal ongeveer 60 mA per led bij vol wit op 100% helderheid.

Formule:

- I_max = aantal_leds * 0.06 A

Voorbeeld bij 191 leds totaal:

- 191 * 0.06 = 11.46 A (theoretisch maximum)

In praktijk vaak lager door brightness-limiet, maar kies voeding met marge.

## Let op software

Niet elke Python library op Raspberry Pi ondersteunt 4 volledig onafhankelijke WS2812B-strips tegelijk.
Als je library limiet geeft op aantal kanalen, dan heb je deze opties:

- wisselen naar een driver/library die meerdere kanalen ondersteunt
- een extern led-controller board gebruiken
- of 1 datalijn splitten naar meerdere strips (dan tonen alle strips hetzelfde patroon)

## HC-SR04 ultrasonische sensors (4x)

Bij gebruik van 4 HC-SR04 sensoren koos ik vaste GPIO pinnen die niet conflicteren met de LED-data pins (18, 13, 12, 19).

- Sensor 1: TRIG -> GPIO17 (pin 11), ECHO -> GPIO27 (pin 13)
- Sensor 2: TRIG -> GPIO22 (pin 15), ECHO -> GPIO4  (pin 7)
- Sensor 3: TRIG -> GPIO23 (pin 16), ECHO -> GPIO5  (pin 29)
- Sensor 4: TRIG -> GPIO24 (pin 18), ECHO -> GPIO6  (pin 31)

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

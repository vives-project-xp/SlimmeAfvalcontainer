#include <Adafruit_NeoPixel.h>

#define PIN_REST    4   // strip1
#define PIN_KARTON  5   // strip2
#define PIN_EXTRA   6   // strip3 (niet gebruikt in afvalkeuze)
#define PIN_PMD     7   // strip4

#define NUM_LEDS     5
#define BRIGHTNESS   50

Adafruit_NeoPixel stripRest(NUM_LEDS,   PIN_REST,   NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripKarton(NUM_LEDS, PIN_KARTON, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripExtra(NUM_LEDS,  PIN_EXTRA,  NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripPmd(NUM_LEDS,    PIN_PMD,    NEO_GRB + NEO_KHZ800);

enum Choice { NONE, PMD, REST, KARTON, ORGANISCH };
Choice currentChoice = NONE;

enum LedMode { MODE_OFF, MODE_SPIN, MODE_BLINK };
LedMode mode = MODE_OFF;

unsigned long lastAnimMs = 0;
const unsigned long animIntervalMs = 90;   // snelheid “cirkel”

// Variabelen voor de knipper-animatie
unsigned long lastBlinkMs = 0;
const unsigned long blinkIntervalMs = 200; // Snelheid van het knipperen (200ms aan, 200ms uit)
int blinkStateCount = 0;                   // Houdt bij in welke stap van de knipper we zitten

int spinIndex = 0;

void setup() {
  Serial.begin(115200);

  stripRest.begin(); stripKarton.begin(); stripExtra.begin(); stripPmd.begin();
  stripRest.setBrightness(BRIGHTNESS);
  stripKarton.setBrightness(BRIGHTNESS);
  stripExtra.setBrightness(BRIGHTNESS);
  stripPmd.setBrightness(BRIGHTNESS);

  allOff();

  Serial.println("Commands:");
  Serial.println("  pmd | rest | karton  -> start spinning green on that strip");
  Serial.println("  hit                -> blink solid green 2x then all off");
  Serial.println("  off                -> all off");
}

void loop() {
  handleSerial();
  updateLeds();
}

void handleSerial() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toLowerCase();

  if (cmd == "pmd") {
    currentChoice = PMD;
    startSpin();
    Serial.println("OK: PMD -> spinning green on pin 7.");
  } 
  else if (cmd == "rest" || cmd == "restafval") {
    currentChoice = REST;
    startSpin();
    Serial.println("OK: REST -> spinning green on pin 4.");
  } 
  else if (cmd == "karton") {
    currentChoice = KARTON;
    startSpin();
    Serial.println("OK: KARTON -> spinning green on pin 5.");
  }
  else if (cmd == "organisch") {
    currentChoice = ORGANISCH;
    startSpin();
    Serial.println("OK: ORGANISCH -> spinning green on pin 6.");
  }
  else if (cmd == "hit") {
    if (currentChoice != NONE) {
      startBlink();
      Serial.println("OK: HIT -> blinking green 2x, then off.");
    } else {
      Serial.println("HIT ignored (no active choice).");
    }
  }
  else if (cmd == "off") {
    currentChoice = NONE;
    mode = MODE_OFF;
    allOff();
    Serial.println("OK: all off.");
  }
  else {
    Serial.println("Unknown. Use: pmd | rest | karton | hit | off");
  }
}

void startSpin() {
  mode = MODE_SPIN;
  spinIndex = 0;
  lastAnimMs = 0;
  allOff();
}

void startBlink() {
  mode = MODE_BLINK;
  blinkStateCount = 0;
  lastBlinkMs = millis();
  showSolidGreen(currentChoice); 
}

void updateLeds() {
  unsigned long now = millis();

  if (mode == MODE_OFF || currentChoice == NONE) {
    return;
  }

  if (mode == MODE_SPIN) {
    if (now - lastAnimMs >= animIntervalMs) {
      lastAnimMs = now;
      showSpinningGreen(currentChoice, spinIndex);
      spinIndex = (spinIndex + 1) % NUM_LEDS;
    }
  }
  else if (mode == MODE_BLINK) {
    if (now - lastBlinkMs >= blinkIntervalMs) {
      lastBlinkMs = now;
      blinkStateCount++;

      if (blinkStateCount >= 8) {
        mode = MODE_OFF;
        currentChoice = NONE;
        allOff();
      } else {

        if (blinkStateCount % 2 == 1) {
          allOff();
        } else {
          showSolidGreen(currentChoice);
        }
      }
    }
  }
}

Adafruit_NeoPixel* getStripForChoice(Choice c) {
  if (c == REST) return &stripRest;     // pin 4
  if (c == KARTON) return &stripKarton; // pin 5
  if (c == PMD) return &stripPmd;       // pin 7
  if (c == ORGANISCH) return &stripExtra; // pin 6
  return nullptr;
}

void allOff() {
  stripRest.clear(); stripRest.show();
  stripKarton.clear(); stripKarton.show();
  stripExtra.clear(); stripExtra.show();
  stripPmd.clear(); stripPmd.show();
}

void showSpinningGreen(Choice c, int idx) {
  // Bepaal de kleuren
  uint32_t GREEN = stripRest.Color(0, 255, 0);
  uint32_t RED   = stripRest.Color(255, 0, 0);

  // Zet eerst alle strips volledig rood
  for (int i = 0; i < NUM_LEDS; i++) {
    stripRest.setPixelColor(i, RED);
    stripKarton.setPixelColor(i, RED);
    stripExtra.setPixelColor(i, RED);
    stripPmd.setPixelColor(i, RED);
  }

  // Wis alleen de gekozen strip (zodat de rest van die specifieke strip uit is) 
  // en zet het draaiende groene lampje aan
  if (c == REST) {
    stripRest.clear();
    stripRest.setPixelColor(idx, GREEN);
  } else if (c == KARTON) {
    stripKarton.clear();
    stripKarton.setPixelColor(idx, GREEN);
  } else if (c == PMD) {
    stripPmd.clear();
    stripPmd.setPixelColor(idx, GREEN);
  } else if (c == ORGANISCH) {
    stripExtra.clear();
    stripExtra.setPixelColor(idx, GREEN);
  }

  // update de strips
  stripRest.show();
  stripKarton.show();
  stripExtra.show();
  stripPmd.show();
}

void showSolidGreen(Choice c) {
  // alles uit
  stripRest.clear();
  stripKarton.clear();
  stripExtra.clear();
  stripPmd.clear();

  uint32_t greenRest = stripRest.Color(0, 255, 0);
  uint32_t greenKarton = stripKarton.Color(0, 255, 0);
  uint32_t greenPmd = stripPmd.Color(0, 255, 0);
  uint32_t greenExtra = stripExtra.Color(0, 255, 0);

  if (c == REST) {
    for (int i = 0; i < NUM_LEDS; i++) stripRest.setPixelColor(i, greenRest);
  } else if (c == KARTON) {
    for (int i = 0; i < NUM_LEDS; i++) stripKarton.setPixelColor(i, greenKarton);
  } else if (c == PMD) {
    for (int i = 0; i < NUM_LEDS; i++) stripPmd.setPixelColor(i, greenPmd);
  } else if (c == ORGANISCH) {
    for (int i = 0; i < NUM_LEDS; i++) stripExtra.setPixelColor(i, greenExtra);
  }

  stripRest.show();
  stripKarton.show();
  stripExtra.show();
  stripPmd.show();
}
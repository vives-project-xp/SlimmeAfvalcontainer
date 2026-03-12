#include <Adafruit_NeoPixel.h>
// Pinnen voor datalijn van ledstrip definen
#define PIN_REST      4   
#define PIN_KARTON    5   
#define PIN_ORGANISCH 6   
#define PIN_PMD       7   

// Aantal leds per afvaltype definen

#define NUM_REST      51
#define NUM_KARTON    38
#define NUM_PMD       51
#define NUM_ORGANISCH 51

// Een brightness van 150 behouden
#define STATIC_BRIGHTNESS 150 

// Alle strips met de library adafruitneopixel meegeven
Adafruit_NeoPixel stripRest(NUM_REST,       PIN_REST,      NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripKarton(NUM_KARTON,   PIN_KARTON,    NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripExtra(NUM_ORGANISCH, PIN_ORGANISCH, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel stripPmd(NUM_PMD,         PIN_PMD,       NEO_GRB + NEO_KHZ800);

uint32_t COLOR_RED   = 0;
uint32_t COLOR_GREEN = 0;
uint32_t COLOR_GREEN_BRIGHT = 0;

enum Choice { NONE, PMD, REST, KARTON, ORGANISCH };
Choice currentChoice = NONE;

enum LedMode { MODE_OFF, MODE_STATIC, MODE_BLINK };
LedMode mode = MODE_OFF;

unsigned long lastBlinkMs = 0;
const unsigned long blinkIntervalMs = 200; 
int blinkStateCount = 0;                   

// Volledige led strip in de gekozen kleur doen
void setStrip(Adafruit_NeoPixel &strip, uint32_t color) {
  for (int i = 0; i < strip.numPixels(); i++) {
    strip.setPixelColor(i, color);
  }
  strip.show();
}

void allOff() {
  setStrip(stripRest,   0);
  setStrip(stripKarton, 0);
  setStrip(stripExtra,  0);
  setStrip(stripPmd,    0);
}

void setup() {
  Serial.begin(115200);

  stripRest.begin();
  stripKarton.begin();
  stripExtra.begin();
  stripPmd.begin();

  COLOR_RED          = stripRest.Color(STATIC_BRIGHTNESS, 0, 0);
  COLOR_GREEN        = stripRest.Color(0, STATIC_BRIGHTNESS, 0);
  COLOR_GREEN_BRIGHT = stripRest.Color(0, 255, 0);

  allOff();
  Serial.println("Systeem Klaar.");
}

void loop() {
  handleSerial();
  updateLeds();
}
// Leest de serial monitor af om de keuze te bepalen
void handleSerial() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toLowerCase();

  if      (cmd == "pmd")                        { currentChoice = PMD;      startStatic(); }
  else if (cmd == "rest")                       { currentChoice = REST;     startStatic(); }
  else if (cmd == "karton" || cmd == "papier")  { currentChoice = KARTON;   startStatic(); }
  else if (cmd == "organisch" || cmd == "bio")  { currentChoice = ORGANISCH;startStatic(); }
  else if (cmd == "hit") {
    if (currentChoice != NONE) startBlink();
  }
  else if (cmd == "off") {
    currentChoice = NONE;
    mode = MODE_OFF;
    allOff();
  }
}

void startStatic() {
  mode = MODE_STATIC;
  updateStrips();
}

void startBlink() {
  mode = MODE_BLINK;
  blinkStateCount = 0;
  lastBlinkMs = millis() - blinkIntervalMs; 
}

void updateStrips() {
  setStrip(stripRest,   (currentChoice == REST)      ? COLOR_GREEN : COLOR_RED);
  setStrip(stripKarton, (currentChoice == KARTON)    ? COLOR_GREEN : COLOR_RED);
  setStrip(stripPmd,    (currentChoice == PMD)       ? COLOR_GREEN : COLOR_RED);
  setStrip(stripExtra,  (currentChoice == ORGANISCH) ? COLOR_GREEN : COLOR_RED);
}
// Gekozen ledstrip groen doen
void showFullGreen(Choice c) {
  setStrip(stripRest,   (c == REST)      ? COLOR_GREEN_BRIGHT : 0);
  setStrip(stripKarton, (c == KARTON)    ? COLOR_GREEN_BRIGHT : 0);
  setStrip(stripPmd,    (c == PMD)       ? COLOR_GREEN_BRIGHT : 0);
  setStrip(stripExtra,  (c == ORGANISCH) ? COLOR_GREEN_BRIGHT : 0);
}

void updateLeds() {
  if (mode == MODE_OFF || currentChoice == NONE) return;

  unsigned long now = millis();

  if (mode == MODE_BLINK) {
    if (now - lastBlinkMs >= blinkIntervalMs) {
      lastBlinkMs = now;
      blinkStateCount++;

      if (blinkStateCount >= 8) {
        mode = MODE_OFF;
        currentChoice = NONE;
        allOff();
      } else if (blinkStateCount % 2 == 1) {
        allOff();
      } else {
        showFullGreen(currentChoice);
      }
    }
  }
}
#include <Adafruit_NeoPixel.h>

#define PIN1 7
#define PIN2 4
#define PIN3 5
#define PIN4 6

#define NUM_LEDS 5
#define BRIGHTNESS 10

Adafruit_NeoPixel strip1(NUM_LEDS, PIN1, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel strip2(NUM_LEDS, PIN2, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel strip3(NUM_LEDS, PIN3, NEO_GRB + NEO_KHZ800);
Adafruit_NeoPixel strip4(NUM_LEDS, PIN4, NEO_GRB + NEO_KHZ800);

enum Choice { NONE, PMD, REST, KARTON };
Choice currentChoice = NONE;

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }

  strip1.begin(); strip2.begin(); strip3.begin(); strip4.begin();
  strip1.setBrightness(BRIGHTNESS);
  strip2.setBrightness(BRIGHTNESS);
  strip3.setBrightness(BRIGHTNESS);
  strip4.setBrightness(BRIGHTNESS);

  clearAll();

  Serial.println("Typ: pmd | restafval | karton");
  Serial.println("Extra: status | off");
}

void loop() {
  handleSerial();

  // Update LEDs volgens keuze
  showChoice(currentChoice);

  delay(50);
}

void handleSerial() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toLowerCase();

  if (cmd == "pmd") {
    currentChoice = PMD;
    Serial.println("OK: PMD geselecteerd (strip op pin 7 wordt GROEN).");
  } 
  else if (cmd == "restafval" || cmd == "rest") {
    currentChoice = REST;
    Serial.println("OK: RESTAFVAL geselecteerd (strip op pin 4 wordt GROEN).");
  } 
  else if (cmd == "karton") {
    currentChoice = KARTON;
    Serial.println("OK: KARTON geselecteerd (strip op pin 5 wordt GROEN).");
  } 
  else if (cmd == "off") {
    currentChoice = NONE;
    clearAll();
    Serial.println("OK: alles uit.");
  }
  else if (cmd == "status") {
    Serial.print("Huidige keuze: ");
    if (currentChoice == PMD) Serial.println("PMD");
    else if (currentChoice == REST) Serial.println("RESTAFVAL");
    else if (currentChoice == KARTON) Serial.println("KARTON");
    else Serial.println("NONE");
  }
  else {
    Serial.println("Onbekend commando. Gebruik: pmd | restafval | karton | status | off");
  }
}

void showChoice(Choice c) {
  uint32_t RED   = strip1.Color(255, 0, 0);
  uint32_t GREEN = strip1.Color(0, 255, 0);
  uint32_t OFF   = strip1.Color(0, 0, 0);

  // Default: alles rood (of uit als NONE)
  uint32_t c1 = (c == NONE) ? OFF : RED;
  uint32_t c2 = (c == NONE) ? OFF : RED;
  uint32_t c3 = (c == NONE) ? OFF : RED;
  uint32_t c4 = (c == NONE) ? OFF : RED;

  // Zet gekozen strip groen
  if (c == PMD)   c1 = GREEN;   // pin 7
  if (c == REST)  c2 = GREEN;   // pin 4
  if (c == KARTON)c3 = GREEN;   // pin 5

  fillStrip(strip1, c1);
  fillStrip(strip2, c2);
  fillStrip(strip3, c3);
  fillStrip(strip4, c4);
}

void fillStrip(Adafruit_NeoPixel &strip, uint32_t color) {
  for (int i = 0; i < NUM_LEDS; i++) strip.setPixelColor(i, color);
  strip.show();
}

void clearAll() {
  strip1.clear(); strip1.show();
  strip2.clear(); strip2.show();
  strip3.clear(); strip3.show();
  strip4.clear(); strip4.show();
}

"""
SlimmeAfvalcontainer – ultrasoon + WS2812B LED-strips (rpi_ws281x backend).

Sensor → strip mapping:
    Sensor 1 – Restafval  → GPIO18  (PWM0, channel 0)
    Sensor 2 – PMD        → GPIO13  (PWM1, channel 1)
    Sensor 3 – Papier     → GPIO12  (PWM0 alt, channel 0)
    Sensor 4 – Organisch       → GPIO19  (PCM, channel 1)

⚠️  rpi_ws281x vereist sudo:
        sudo python slimme_afvalcontainer.py
"""

from __future__ import annotations

import time
import threading

import RPi.GPIO as GPIO
from rpi_ws281x import PixelStrip, Color, WS2811_STRIP_GRB


# ── Ultrasoon configuratie ────────────────────────────────────────────────────
SENSOR_1_TRIG = 25;  SENSOR_1_ECHO = 26
SENSOR_2_TRIG = 22;  SENSOR_2_ECHO = 4
SENSOR_3_TRIG = 23;  SENSOR_3_ECHO = 24
SENSOR_4_TRIG = 17;  SENSOR_4_ECHO = 27

CONTAINER_HEIGHT_CM = 65.0
DEBOUNCE_TIME       = 1.5

# HC-SR04 timing constants
TRIGGER_PULSE_S     = 0.00001   # 10 µs trigger pulse
ECHO_TIMEOUT_S      = 0.04      # 40 ms → max ~6.9 m range; more forgiving than 25ms
INTER_SAMPLE_S      = 0.070     # 70 ms between samples (HC-SR04 datasheet)
INTER_SENSOR_S      = 0.05      # 50 ms between sensors to avoid crosstalk

SENSORS = [
    {"id": 1, "name": "Restafval", "strip_idx": 0, "trig": SENSOR_1_TRIG, "echo": SENSOR_1_ECHO, "threshold": 5},
    {"id": 2, "name": "PMD",       "strip_idx": 1, "trig": SENSOR_2_TRIG, "echo": SENSOR_2_ECHO, "threshold": 5},
    {"id": 3, "name": "Papier",    "strip_idx": 2, "trig": SENSOR_3_TRIG, "echo": SENSOR_3_ECHO, "threshold": 5},
    {"id": 4, "name": "Glas",      "strip_idx": 3, "trig": SENSOR_4_TRIG, "echo": SENSOR_4_ECHO, "threshold": 5},
]


# ── LED-strip configuratie ────────────────────────────────────────────────────
STRIP_CONFIGS = [
    {"name": "Restafval", "pin": 18, "count": 51, "channel": 0, "dma": 10},
    {"name": "PMD",       "pin": 13, "count": 51, "channel": 1, "dma": 11},
    {"name": "Papier",    "pin": 12, "count": 38, "channel": 0, "dma": 12},
    {"name": "Glas",      "pin": 19, "count": 51, "channel": 1, "dma": 13},
]

BASE_COLORS = [
    (255,  70,  70),   # Restafval – rood
    (255, 190,  40),   # PMD       – geel
    ( 70, 140, 255),   # Papier    – blauw
    ( 40, 190, 170),   # Glas      – teal
]

ORANGE = (255, 100,   0)
WHITE  = (180, 180, 180)
OFF    = (  0,   0,   0)

BRIGHTNESS = 128  # 0–255


# ── Strip controller ──────────────────────────────────────────────────────────
def _col(rgb: tuple) -> int:
    return Color(rgb[0], rgb[1], rgb[2])


class StripController:
    def __init__(self):
        self._strips: list[PixelStrip] = []
        self._lock = threading.Lock()
        self.enabled = False

        try:
            for cfg in STRIP_CONFIGS:
                strip = PixelStrip(
                    cfg["count"], cfg["pin"],
                    freq_hz=800_000,
                    dma=cfg["dma"],
                    invert=False,
                    brightness=BRIGHTNESS,
                    channel=cfg["channel"],
                    strip_type=WS2811_STRIP_GRB,
                )
                strip.begin()
                self._strips.append(strip)

            self._fill_all(OFF)
            self.enabled = True
            print(f"[LED] {len(self._strips)} strips actief.")
        except Exception as exc:
            print(f"[LED] Init mislukt: {exc}\n  → Draai het script met sudo!")
            self._strips.clear()

    def _fill(self, idx: int, rgb: tuple) -> None:
        strip = self._strips[idx]
        c = _col(rgb)
        for i in range(strip.numPixels()):
            strip.setPixelColor(i, c)
        strip.show()

    def _fill_all(self, rgb: tuple) -> None:
        for i in range(len(self._strips)):
            self._fill(i, rgb)

    def set_fill_color(self, idx: int, fill_pct: int) -> None:
        if not self.enabled:
            return
        r, g, b = BASE_COLORS[idx]
        if fill_pct < 0:
            rgb = OFF
        elif fill_pct < 50:
            rgb = (int(r * 0.4), int(g * 0.4), int(b * 0.4))
        elif fill_pct < 80:
            rgb = (int(r * 0.7), int(g * 0.7), int(b * 0.7))
        else:
            rgb = (r, g, b)
        with self._lock:
            self._fill(idx, rgb)

    def blink(self, idx: int, rgb: tuple, times: int = 3, interval: float = 0.15) -> None:
        if not self.enabled:
            return
        with self._lock:
            for _ in range(times):
                self._fill(idx, rgb);  time.sleep(interval)
                self._fill(idx, OFF);  time.sleep(interval)

    def set_color(self, idx: int, rgb: tuple) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._fill(idx, rgb)

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._fill_all(OFF)
        self.enabled = False


# ── GPIO / ultrasoon ──────────────────────────────────────────────────────────
def setup_gpio() -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for s in SENSORS:
        GPIO.setup(s["trig"], GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(s["echo"], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    time.sleep(0.5)
    print("[GPIO] Sensoren geïnitialiseerd.")
    print("[GPIO] Wacht tot sensoren stabiel zijn...")


def check_echo_pins() -> None:
    print("Echo pin diagnose (moet LOW zijn voor alle sensoren):")
    for s in SENSORS:
        state = GPIO.input(s["echo"])
        label = "OK" if state == 0 else "FOUT – pin staat HOOG (los draad of zwevend?)"
        print(f"  GPIO{s['echo']:2d}  {s['name']:10s}: {'LOW' if state == 0 else 'HIGH'}  {label}")


def measure_distance(sensor: dict, retries: int = 2) -> float | None:
    trig = sensor["trig"]
    echo = sensor["echo"]

    for attempt in range(retries):
        try:
            # Guarantee TRIG is LOW before we start
            GPIO.output(trig, GPIO.LOW)
            time.sleep(0.00001)  # 10 µs settling

            # Send 10 µs trigger pulse
            GPIO.output(trig, GPIO.HIGH)
            time.sleep(TRIGGER_PULSE_S)
            GPIO.output(trig, GPIO.LOW)

            # Wait for echo to go HIGH (sensor starts returning pulse)
            timeout_high = time.time() + ECHO_TIMEOUT_S
            pulse_start = None
            while time.time() < timeout_high:
                if GPIO.input(echo) == 1:
                    pulse_start = time.time()
                    break
                time.sleep(0.000001)

            if pulse_start is None:
                continue  # Retry if no echo detected

            # Wait for echo to go LOW (pulse finished)
            timeout_low = time.time() + ECHO_TIMEOUT_S
            pulse_end = None
            while time.time() < timeout_low:
                if GPIO.input(echo) == 0:
                    pulse_end = time.time()
                    break
                time.sleep(0.000001)

            if pulse_end is None:
                continue  # Retry if echo stuck HIGH

            distance_cm = (pulse_end - pulse_start) * 34300 / 2

            # Sanity check: HC-SR04 is reliable in 2–400 cm range
            if 2.0 < distance_cm < 400.0:
                return distance_cm
        except Exception as e:
            print(f"  [Sensor {sensor['id']}] Attempt {attempt + 1} failed: {e}")
            continue

    return None


def measure_average(sensor: dict, samples: int = 3) -> float | None:
    readings: list[float] = []
    for i in range(samples):
        d = measure_distance(sensor, retries=2)
        if d is not None:
            readings.append(d)
        if i < samples - 1:  # Don't sleep after last sample
            time.sleep(INTER_SAMPLE_S)

    if not readings:
        return None

    readings.sort()
    mid = len(readings) // 2
    return readings[mid] if len(readings) % 2 else (readings[mid - 1] + readings[mid]) / 2


def calc_fill_pct(distance: float | None) -> int:
    if distance is None:
        return -1
    pct = ((CONTAINER_HEIGHT_CM - distance) / CONTAINER_HEIGHT_CM) * 100
    return max(0, min(100, int(pct)))


def check_disposed(sensor: dict, prev: float | None, curr: float | None):
    if curr is None or prev is None:
        return False, curr if curr is not None else prev
    if prev - curr >= sensor["threshold"]:
        time.sleep(0.8)
        confirmed = measure_average(sensor, samples=5)
        if confirmed is not None and prev - confirmed >= sensor["threshold"]:
            return True, confirmed
    return False, curr


# ── Hoofdlus ─────────────────────────────────────────────────────────────────
def main() -> None:
    setup_gpio()
    check_echo_pins()
    leds = StripController()
    prev_distances: list[float | None] = []
    for sensor in SENSORS:
        prev_distances.append(measure_average(sensor, samples=5))
        time.sleep(INTER_SENSOR_S)

    last_throw = [0.0] * len(SENSORS)
    is_full    = [False] * len(SENSORS)

    for i, sensor in enumerate(SENSORS):
        leds.set_fill_color(sensor["strip_idx"], calc_fill_pct(prev_distances[i]))

    print("Smart bin klaar!\n")

    try:
        consecutive_failures = [0] * len(SENSORS)

        while True:
            for i, sensor in enumerate(SENSORS):
                idx = sensor["strip_idx"]
                now = time.time()
                if now - last_throw[i] < DEBOUNCE_TIME:
                    continue

                curr = measure_average(sensor, samples=3)

                if curr is None:
                    consecutive_failures[i] += 1
                    if consecutive_failures[i] >= 3:
                        print(f"⚠️  Sensor {sensor['id']} ({sensor['name']}): Herhaalde fouten ({consecutive_failures[i]}x)")
                    else:
                        print(f"[Sensor {sensor['id']}] geen echo (poging {consecutive_failures[i]}/3).")
                    time.sleep(INTER_SENSOR_S)
                    continue

                consecutive_failures[i] = 0
                fill = calc_fill_pct(curr)
                print(f"Sensor {sensor['id']} ({sensor['name']}): {curr:.1f} cm | {fill}%")

                # Vol?
                if curr < 5 or fill >= 95:
                    if not is_full[i]:
                        is_full[i] = True
                        print(f"⚠️  {sensor['name']} is VOL!")
                        leds.blink(idx, ORANGE, times=5, interval=0.1)
                    leds.set_color(idx, ORANGE)
                    prev_distances[i] = curr
                    time.sleep(INTER_SENSOR_S)
                    continue
                else:
                    is_full[i] = False

                # Afval gegooid?
                disposed, updated = check_disposed(sensor, prev_distances[i], curr)
                if disposed:
                    last_throw[i] = time.time()
                    fill_after = calc_fill_pct(updated)
                    print(f"✅ Afval in {sensor['name']} ({prev_distances[i]:.1f} → {updated:.1f} cm) | {fill_after}%")
                    leds.blink(idx, WHITE, times=2, interval=0.1)
                    leds.set_fill_color(idx, fill_after)
                    prev_distances[i] = updated
                else:
                    leds.set_fill_color(idx, fill)
                    prev_distances[i] = curr

                time.sleep(INTER_SENSOR_S)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nAfgesloten.")
    finally:
        leds.close()
        GPIO.cleanup()


if __name__ == "__main__":
    main()

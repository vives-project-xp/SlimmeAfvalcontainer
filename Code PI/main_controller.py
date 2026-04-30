"""
SlimmeAfvalcontainer – ultrasoon + LED-strips (neopixel backend).

Sensor → strip mapping (BCM):
    Restafval  → TRIG 25 / ECHO 26  → GPIO18  (51 LEDs, rood)
    PMD        → TRIG 22 / ECHO  4  → GPIO13  (38 LEDs, geel)
    Papier     → TRIG 23 / ECHO 24  → GPIO12  (51 LEDs, blauw)
    Glas       → TRIG 17 / ECHO 27  → GPIO19  (51 LEDs, teal)

Vullingsgedrag:
    < 50 %  → basiskleur @ 40 % helderheid
    50–79 % → basiskleur @ 70 % helderheid
    ≥ 80 %  → basiskleur volledig
    VOL     → oranje flikker (5×) + blijft oranje
    Afval gegooid → korte witte flits, daarna terug naar vullingskleur
"""

from __future__ import annotations

import time
import board
import neopixel
import RPi.GPIO as GPIO


# ── Configuratie ──────────────────────────────────────────────────────────────

CONTAINER_HEIGHT_CM = 65.0
DEBOUNCE_TIME       = 1.5
BRIGHTNESS          = 0.25

ORANGE = (255, 100,   0)
WHITE  = (180, 180, 180)
OFF    = (  0,   0,   0)

SENSORS = [
    {"name": "Restafval", "trig": 25, "echo": 26, "threshold": 10,
     "pin": board.D18, "count": 51, "color": (255,  70,  70)},
    {"name": "PMD",       "trig": 22, "echo":  4, "threshold": 10,
     "pin": board.D13, "count": 38, "color": (255, 190,  40)},
    {"name": "Papier",    "trig": 23, "echo": 24, "threshold": 10,
     "pin": board.D12, "count": 51, "color": ( 70, 140, 255)},
    {"name": "Glas",      "trig": 17, "echo": 27, "threshold": 10,
     "pin": board.D19, "count": 51, "color": ( 40, 190, 170)},
]


# ── LED helpers ───────────────────────────────────────────────────────────────

def make_strips() -> list[neopixel.NeoPixel]:
    result = []
    for s in SENSORS:
        strip = neopixel.NeoPixel(
            s["pin"], s["count"],
            brightness=BRIGHTNESS,
            auto_write=False,
            pixel_order=neopixel.GRB,
        )
        result.append(strip)
    print(f"[LED] {len(result)} strips actief.")
    return result


def led_set(strips: list, idx: int, rgb: tuple) -> None:
    strips[idx].fill(rgb)
    strips[idx].show()


def led_blink(strips: list, idx: int, rgb: tuple, times: int = 5, interval: float = 0.1) -> None:
    for _ in range(times):
        led_set(strips, idx, rgb)
        time.sleep(interval)
        led_set(strips, idx, OFF)
        time.sleep(interval)


def led_fill_color(strips: list, idx: int, fill_pct: int) -> None:
    r, g, b = SENSORS[idx]["color"]
    if fill_pct < 0:
        led_set(strips, idx, OFF)
    elif fill_pct < 50:
        led_set(strips, idx, (int(r * 0.4), int(g * 0.4), int(b * 0.4)))
    elif fill_pct < 80:
        led_set(strips, idx, (int(r * 0.7), int(g * 0.7), int(b * 0.7)))
    else:
        led_set(strips, idx, (r, g, b))


def led_close(strips: list) -> None:
    for strip in strips:
        strip.fill(OFF)
        strip.show()
        strip.deinit()


# ── Ultrasoon helpers ─────────────────────────────────────────────────────────

def setup_gpio() -> None:
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for s in SENSORS:
        GPIO.setup(s["trig"], GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(s["echo"], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    time.sleep(0.5)


def measure_distance(sensor: dict, timeout: float = 0.038) -> float | None:
    t = time.time()
    while GPIO.input(sensor["echo"]) == 1:
        if time.time() - t >= timeout:
            return None

    GPIO.output(sensor["trig"], GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(sensor["trig"], GPIO.LOW)

    t = time.time()
    while GPIO.input(sensor["echo"]) == 0:
        if time.time() - t >= timeout:
            return None
    start = time.time()

    while GPIO.input(sensor["echo"]) == 1:
        if time.time() - start >= timeout:
            return None
    stop = time.time()

    distance = ((stop - start) * 34300) / 2
    return distance if 2 < distance <= 400 else None


def measure_average(sensor: dict, samples: int = 3) -> float | None:
    readings = []
    for _ in range(samples):
        d = measure_distance(sensor)
        if d is not None:
            readings.append(d)
        time.sleep(0.03)
    return sum(readings) / len(readings) if readings else None


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


# ── Hoofdlus ──────────────────────────────────────────────────────────────────

def main() -> None:
    setup_gpio()
    strips = make_strips()

    prev_distances = [measure_average(s, samples=5) for s in SENSORS]
    last_throw     = [0.0] * len(SENSORS)
    is_full        = [False] * len(SENSORS)

    for i in range(len(SENSORS)):
        led_fill_color(strips, i, calc_fill_pct(prev_distances[i]))

    print("Smart bin klaar!\n")

    try:
        while True:
            for i, sensor in enumerate(SENSORS):
                now = time.time()
                if now - last_throw[i] < DEBOUNCE_TIME:
                    continue

                curr = measure_average(sensor, samples=3)

                if curr is None:
                    print(f"Sensor {sensor['name']}: geen echo.")
                    led_set(strips, i, OFF)
                    continue

                fill = calc_fill_pct(curr)
                print(f"Sensor {sensor['name']}: {curr:.1f} cm | {fill}%")

                if curr < 5 or fill >= 95:
                    if not is_full[i]:
                        is_full[i] = True
                        print(f"  {sensor['name']} is VOL!")
                        led_blink(strips, i, ORANGE, times=5, interval=0.1)
                    led_set(strips, i, ORANGE)
                    prev_distances[i] = curr
                    continue

                is_full[i] = False

                disposed, updated = check_disposed(sensor, prev_distances[i], curr)
                if disposed:
                    last_throw[i] = time.time()
                    fill_after = calc_fill_pct(updated)
                    print(f"Afval in {sensor['name']} ({prev_distances[i]:.1f} -> {updated:.1f} cm) | {fill_after}%")
                    led_blink(strips, i, WHITE, times=2, interval=0.1)
                    led_fill_color(strips, i, fill_after)
                    prev_distances[i] = updated
                else:
                    led_fill_color(strips, i, fill)
                    prev_distances[i] = curr

                time.sleep(0.06)

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nAfgesloten.")
    finally:
        led_close(strips)
        GPIO.cleanup()


if __name__ == "__main__":
    main()

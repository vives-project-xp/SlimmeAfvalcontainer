"""GPIO controller voor 4 WS2812B strips direct op Raspberry Pi.

Pin mapping (BCM):
    REST      -> GPIO18
    KARTON    -> GPIO13
    ORGANISCH -> GPIO12
    PMD       -> GPIO19

Deze module houdt dezelfde `send_command(cmd)` API aan als de oude UART-versie,
zodat hij direct gebruikt kan worden door `inference_gui.py`.
"""

from __future__ import annotations

import os
import threading
from enum import Enum, auto

import board
import neopixel

# ── Configuratie ─────────────────────────────────────────────────────────────
# Pas deze aan op basis van het aantal leds per strip.
# Huidige hardware:
# - REST: 51
# - KARTON: 38
# - ORGANISCH: 51
# - PMD: 51
DEFAULT_BRIGHTNESS = float(os.getenv("LED_BRIGHTNESS", "0.25"))
MAX_SUPPLY_CURRENT_A = float(os.getenv("LED_MAX_CURRENT_A", "5.0"))
CURRENT_HEADROOM = float(os.getenv("LED_CURRENT_HEADROOM", "0.85"))

PIN_MAP = {
        "rest": board.D18,
        "karton": board.D13,
        "organisch": board.D12,
        "pmd": board.D19,
}

DEFAULT_LED_COUNT_MAP = {
    "rest": int(os.getenv("LED_COUNT_REST", "51")),
    "karton": int(os.getenv("LED_COUNT_KARTON", "38")),
    "organisch": int(os.getenv("LED_COUNT_ORGANISCH", "51")),
    "pmd": int(os.getenv("LED_COUNT_PMD", "51")),
}

COLOR_MAP = {
        "rest": (255, 70, 70),
        "karton": (70, 140, 255),
        "organisch": (60, 220, 90),
        "pmd": (255, 190, 40),
}

OFF = (0, 0, 0)

class Choice(Enum):
    NONE      = auto()
    PMD       = auto()
    REST      = auto()
    KARTON    = auto()
    ORGANISCH = auto()


class LedController:
    """
    Stuurt commando's naar 4 WS2812B strips op GPIO.

    Methode:
        send_command(cmd: str) -> str
            Accepteert: "pmd", "rest", "karton", "papier",
                        "organisch", "bio", "hit", "off", "reset".
            Stuurt leds aan en geeft een status-string terug.
    """

    def __init__(self, led_count_map: dict[str, int] | None = None, brightness: float = DEFAULT_BRIGHTNESS):
        self.enabled = False
        self.current_choice = Choice.NONE
        self._lock = threading.Lock()
        self._strips: dict[str, neopixel.NeoPixel] = {}
        self._led_count_map: dict[str, int] = {}
        self._max_allowed_current_a = max(0.1, MAX_SUPPLY_CURRENT_A * CURRENT_HEADROOM)

        if led_count_map is None:
            led_count_map = DEFAULT_LED_COUNT_MAP.copy()

        try:
            for name, pin in PIN_MAP.items():
                led_count = int(led_count_map.get(name, 1))
                if led_count < 1:
                    led_count = 1
                self._led_count_map[name] = led_count
                self._strips[name] = neopixel.NeoPixel(
                    pin,
                    led_count,
                    brightness=brightness,
                    auto_write=False,
                    pixel_order=neopixel.GRB,
                )

            self._set_all(OFF)
            self.enabled = True
            print("[LED] Controller gereed. 4 strips actief op GPIO 18/13/12/19")
            print(
                f"[LED] Current limiter actief: max {self._max_allowed_current_a:.2f}A "
                f"(supply={MAX_SUPPLY_CURRENT_A:.2f}A, headroom={CURRENT_HEADROOM:.2f})"
            )
        except Exception as exc:
            print(
                f"[LED] Waarschuwing: GPIO/NeoPixel init mislukt: {exc}\n"
                "  Controleer wiring, permissies en rpi_ws281x setup.\n"
                "  LED-aansturing wordt overgeslagen."
            )

    # ── Publieke interface ────────────────────────────────────────────────────

    def _set_all(self, rgb: tuple[int, int, int]) -> None:
        for strip in self._strips.values():
            strip.fill(rgb)
            strip.show()

    @staticmethod
    def _estimate_led_current_a(rgb: tuple[int, int, int]) -> float:
        # WS2812B kanaalstroom benadering: ~20mA per kanaal op 255.
        r, g, b = rgb
        return ((r + g + b) / 255.0) * 0.02

    def _limit_color_for_strip(self, strip_name: str, rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        led_count = self._led_count_map.get(strip_name, 1)
        estimated_total = self._estimate_led_current_a(rgb) * led_count
        if estimated_total <= self._max_allowed_current_a:
            return rgb

        # Schaal RGB proportioneel zodat I_est <= limiet.
        scale = self._max_allowed_current_a / max(estimated_total, 1e-6)
        r, g, b = rgb
        limited = (
            max(0, min(255, int(r * scale))),
            max(0, min(255, int(g * scale))),
            max(0, min(255, int(b * scale))),
        )
        print(
            f"[LED] Current limit toegepast voor {strip_name}: "
            f"{rgb} -> {limited} (est {estimated_total:.2f}A > {self._max_allowed_current_a:.2f}A)"
        )
        return limited

    def _show_bin(self, bin_name: str) -> None:
        self._set_all(OFF)
        strip = self._strips.get(bin_name)
        if strip is not None:
            color = self._limit_color_for_strip(bin_name, COLOR_MAP[bin_name])
            strip.fill(color)
            strip.show()

    def send_command(self, cmd: str) -> str:
        """
        Verwerk een commando-string en stuur de strips aan.

        Geeft een status-string terug.
        """
        cmd = cmd.strip().lower()

        # Alias voor compatibiliteit met oudere flow.
        if cmd == "papier":
            cmd = "karton"
        elif cmd == "bio":
            cmd = "organisch"

        if not self.enabled:
            print(f"[LED] (uitgeschakeld) commando: {cmd}")
            return f"OK: {cmd.upper()}"

        with self._lock:
            try:
                if cmd in PIN_MAP:
                    self._show_bin(cmd)
                    self.current_choice = {
                        "pmd": Choice.PMD,
                        "rest": Choice.REST,
                        "karton": Choice.KARTON,
                        "organisch": Choice.ORGANISCH,
                    }[cmd]
                    return f"OK: {cmd.upper()}"

                if cmd == "hit":
                    # Herhaal laatste keuze.
                    mapping = {
                        Choice.PMD: "pmd",
                        Choice.REST: "rest",
                        Choice.KARTON: "karton",
                        Choice.ORGANISCH: "organisch",
                    }
                    selected = mapping.get(self.current_choice)
                    if selected:
                        self._show_bin(selected)
                    return "OK: HIT"

                if cmd in {"off", "reset"}:
                    self._set_all(OFF)
                    if cmd == "reset":
                        self.current_choice = Choice.NONE
                    return f"OK: {cmd.upper()}"

                return f"UNKNOWN: {cmd}"
            except Exception as exc:
                print(f"[LED] Fout bij commando '{cmd}': {exc}")
                return f"ERROR: {cmd}"

    def close(self) -> None:
        """Zet alle leds uit en geef resources vrij."""
        if not self.enabled:
            return

        with self._lock:
            try:
                self._set_all(OFF)
            finally:
                for strip in self._strips.values():
                    try:
                        strip.deinit()
                    except Exception:
                        pass
                self._strips.clear()
                self.enabled = False


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    ctrl = LedController()
    for cmd in ["pmd", "rest", "karton", "organisch", "hit", "off"]:
        print(f"Verzonden: {cmd} -> {ctrl.send_command(cmd)}")
        time.sleep(2)
    ctrl.close()


"""LED-strip controller voor Raspberry Pi — vervangt de ESP32.

Stuurt 4 WS2812B strips aan via rpi_ws281x (PWM/DMA).
Gebruikt 2 PWM-kanalen met 2 strips per kanaal (daisy-chain):

  Kanaal 0 → GPIO 18 (header pin 12)
      REST(51 LEDs) output → ORGANISCH(51 LEDs) input

  Kanaal 1 → GPIO 13 (header pin 33)
      KARTON(38 LEDs) output → PMD(51 LEDs) input

GND van alle strips → Pi GND  (pin 6, 9, 14, ...)
Voeding strips      → externe 5V (NIET de Pi 5V pin)

Draaien vereist root (sudo) vanwege DMA/PWM-toegang:
  sudo -E <venv>/bin/python led_controller.py
"""

import threading
import time
from enum import Enum, auto

# ── Configuratie ─────────────────────────────────────────────────────────────
NUM_REST       = 51
NUM_KARTON     = 38
NUM_PMD        = 51
NUM_ORGANISCH  = 51

STATIC_BRIGHTNESS = 150    # 0–255
BLINK_INTERVAL_S  = 0.2    # seconden per halve knipper
BLINK_CYCLES      = 4      # aantal volledige aan/uit-cycli

# PWM-kanalen
PIN_CH0  = 18              # GPIO 18, header pin 12  (PWM channel 0)
PIN_CH1  = 13              # GPIO 13, header pin 33  (PWM channel 1)
DMA_CH0  = 10
DMA_CH1  = 11

# Pixel-indices binnen elke keten
IDX_REST_START       = 0
IDX_REST_END         = NUM_REST                    # 51
IDX_ORGANISCH_START  = NUM_REST                    # 51
IDX_ORGANISCH_END    = NUM_REST + NUM_ORGANISCH    # 102

IDX_KARTON_START     = 0
IDX_KARTON_END       = NUM_KARTON                  # 38
IDX_PMD_START        = NUM_KARTON                  # 38
IDX_PMD_END          = NUM_KARTON + NUM_PMD        # 89

# Kleuren (R, G, B)
COLOR_RED          = (STATIC_BRIGHTNESS, 0, 0)
COLOR_GREEN        = (0, STATIC_BRIGHTNESS, 0)
COLOR_GREEN_BRIGHT = (0, 255, 0)
COLOR_OFF          = (0, 0, 0)
# ─────────────────────────────────────────────────────────────────────────────


class Choice(Enum):
    NONE      = auto()
    PMD       = auto()
    REST      = auto()
    KARTON    = auto()
    ORGANISCH = auto()


class LedController:
    """
    Stuurt de 4 ledstrips aan zoals de ESP32 deed.

    Methode:
        send_command(cmd: str) -> str
            Accepteert: "pmd", "rest", "karton", "papier",
                        "organisch", "bio", "hit", "off", "reset"
            Geeft terug: bevestigingsstring (zelfde als ESP32's Serial1.println)
    """

    def __init__(self):
        self.current_choice = Choice.NONE
        self._lock   = threading.Lock()
        self._blinking       = False
        self._blink_thread: threading.Thread | None = None
        self._strip0 = None   # kanaal 0: REST(0-50) + ORGANISCH(51-101)
        self._strip1 = None   # kanaal 1: KARTON(0-37) + PMD(38-88)
        self._Color  = None

        try:
            from rpi_ws281x import PixelStrip, Color, ws
            self._Color = Color
            strip_type  = ws.WS2811_STRIP_GRB

            self._strip0 = PixelStrip(
                IDX_ORGANISCH_END, PIN_CH0,
                800000, DMA_CH0, False, 255, 0, strip_type
            )
            self._strip0.begin()

            self._strip1 = PixelStrip(
                IDX_PMD_END, PIN_CH1,
                800000, DMA_CH1, False, 255, 1, strip_type
            )
            self._strip1.begin()

            self.enabled = True
            self.all_off()
            print(f"[LED] Controller gereed. CH0=GPIO{PIN_CH0} (REST+ORGANISCH), "
                  f"CH1=GPIO{PIN_CH1} (KARTON+PMD)")

        except Exception as exc:
            self.enabled = False
            print(
                f"[LED] Waarschuwing: rpi_ws281x init mislukt: {exc}\n"
                "  Controleer sudo-rechten en GPIO-aansluitingen.\n"
                "  LED-aansturing wordt overgeslagen."
            )

    # ── Interne helpers ───────────────────────────────────────────────────────

    def _fill_strip0(self, rest_rgb: tuple, organisch_rgb: tuple) -> None:
        """Vul kanaal 0: REST (indices 0-50) en ORGANISCH (indices 51-101)."""
        c_rest = self._Color(*rest_rgb)
        c_org  = self._Color(*organisch_rgb)
        for i in range(IDX_REST_START, IDX_REST_END):
            self._strip0.setPixelColor(i, c_rest)
        for i in range(IDX_ORGANISCH_START, IDX_ORGANISCH_END):
            self._strip0.setPixelColor(i, c_org)
        self._strip0.show()

    def _fill_strip1(self, karton_rgb: tuple, pmd_rgb: tuple) -> None:
        """Vul kanaal 1: KARTON (indices 0-37) en PMD (indices 38-88)."""
        c_kar = self._Color(*karton_rgb)
        c_pmd = self._Color(*pmd_rgb)
        for i in range(IDX_KARTON_START, IDX_KARTON_END):
            self._strip1.setPixelColor(i, c_kar)
        for i in range(IDX_PMD_START, IDX_PMD_END):
            self._strip1.setPixelColor(i, c_pmd)
        self._strip1.show()

    def all_off(self) -> None:
        if not self.enabled:
            return
        self._fill_strip0(COLOR_OFF, COLOR_OFF)
        self._fill_strip1(COLOR_OFF, COLOR_OFF)

    def _update_static(self) -> None:
        """Gekozen strip groen, rest rood — zelfde als ESP32's updateStrips()."""
        c = self.current_choice
        self._fill_strip0(
            COLOR_GREEN if c == Choice.REST      else COLOR_RED,
            COLOR_GREEN if c == Choice.ORGANISCH else COLOR_RED,
        )
        self._fill_strip1(
            COLOR_GREEN if c == Choice.KARTON    else COLOR_RED,
            COLOR_GREEN if c == Choice.PMD       else COLOR_RED,
        )

    def _show_full_green(self, choice: Choice) -> None:
        """Alleen gekozen strip fel groen, rest uit — zelfde als ESP32's showFullGreen()."""
        off = COLOR_OFF
        fg  = COLOR_GREEN_BRIGHT
        self._fill_strip0(
            fg  if choice == Choice.REST      else off,
            fg  if choice == Choice.ORGANISCH else off,
        )
        self._fill_strip1(
            fg  if choice == Choice.KARTON    else off,
            fg  if choice == Choice.PMD       else off,
        )

    def _blink_worker(self, choice: Choice) -> None:
        """Achtergrond-thread: knippert de gekozen strip 4× (= ESP32 MODE_BLINK)."""
        for _ in range(BLINK_CYCLES):
            if not self._blinking:
                break
            self.all_off()
            time.sleep(BLINK_INTERVAL_S)
            if not self._blinking:
                break
            self._show_full_green(choice)
            time.sleep(BLINK_INTERVAL_S)
        self.all_off()
        with self._lock:
            self._blinking = False
            self.current_choice = Choice.NONE

    def _stop_blink(self) -> None:
        """Stopt een lopende knipperthread. Moet aangeroepen worden mét de lock."""
        if self._blinking:
            self._blinking = False
            # De thread controleert _blinking en stopt vanzelf.

    # ── Publieke interface ────────────────────────────────────────────────────

    def send_command(self, cmd: str) -> str:
        """
        Verwerk een commando-string — zelfde interface als ESP32's handleSerial().

        Geeft een bevestigingsstring terug die logbaar is via print().
        """
        cmd = cmd.strip().lower()

        if not self.enabled:
            print(f"[LED] (uitgeschakeld) commando: {cmd}")
            return f"OK: {cmd.upper()}"

        if cmd == "pmd":
            with self._lock:
                self._stop_blink()
                self.current_choice = Choice.PMD
            self._update_static()
            return "OK: PMD"

        elif cmd == "rest":
            with self._lock:
                self._stop_blink()
                self.current_choice = Choice.REST
            self._update_static()
            return "OK: REST"

        elif cmd in ("karton", "papier"):
            with self._lock:
                self._stop_blink()
                self.current_choice = Choice.KARTON
            self._update_static()
            return "OK: KARTON"

        elif cmd in ("organisch", "bio"):
            with self._lock:
                self._stop_blink()
                self.current_choice = Choice.ORGANISCH
            self._update_static()
            return "OK: ORGANISCH"

        elif cmd == "hit":
            with self._lock:
                if self.current_choice == Choice.NONE:
                    return "IGNORED: geen keuze actief"
                self._stop_blink()
                choice = self.current_choice
                self._blinking = True
            t = threading.Thread(target=self._blink_worker, args=(choice,), daemon=True)
            self._blink_thread = t
            t.start()
            return "OK: HIT"

        elif cmd in ("off", "reset"):
            with self._lock:
                self._stop_blink()
                self.current_choice = Choice.NONE
            self.all_off()
            return "OK: OFF"

        return f"ONBEKEND: {cmd}"

    def close(self) -> None:
        """Schakel alle strips uit en ruim op."""
        with self._lock:
            self._stop_blink()
        time.sleep(BLINK_INTERVAL_S * 2)   # geef blink-thread kans te stoppen
        self.all_off()


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ctrl = LedController()
    for cmd in ["pmd", "rest", "karton", "organisch", "hit", "reset"]:
        print(ctrl.send_command(cmd))
        time.sleep(2)
    ctrl.close()

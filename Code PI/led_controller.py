"""UART controller voor Raspberry Pi — stuurt commando's naar ESP32.

De ESP32 luistert op UART1 (RX=GPIO4, TX=GPIO5).
Raspberry Pi 5 UART0: TX=GPIO14 naar ESP32 RX=GPIO4, RX=GPIO15 naar ESP32 TX=GPIO5.

Draaien vereist toegang tot /dev/ttyAMA0 (UART0):
  sudo -E <venv>/bin/python led_controller.py
"""

import serial
from enum import Enum, auto

# ── Configuratie ─────────────────────────────────────────────────────────────
UART_PORT = '/dev/ttyAMA0'  # Raspberry Pi 5 UART0
BAUD_RATE = 115200
TIMEOUT   = 1.0  # seconden voor read timeout

class Choice(Enum):
    NONE      = auto()
    PMD       = auto()
    REST      = auto()
    KARTON    = auto()
    ORGANISCH = auto()


class LedController:
    """
    Stuurt commando's naar de ESP32 via UART.

    Methode:
        send_command(cmd: str) -> str
            Accepteert: "pmd", "rest", "karton", "papier",
                        "organisch", "bio", "hit", "off", "reset"
            Stuurt naar ESP32 en geeft de response terug.
    """

    def __init__(self):
        self.ser = None
        self.enabled = False
        self.current_choice = Choice.NONE

        try:
            self.ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=TIMEOUT)
            self.enabled = True
            print(f"[UART] Controller gereed. Stuurt naar ESP32 via {UART_PORT}")
        except Exception as exc:
            print(
                f"[UART] Waarschuwing: serial init mislukt: {exc}\n"
                "  Controleer UART-aansluitingen en rechten.\n"
                "  UART-communicatie wordt overgeslagen."
            )

    # ── Publieke interface ────────────────────────────────────────────────────

    def send_command(self, cmd: str) -> str:
        """
        Verwerk een commando-string — stuurt naar ESP32.

        Geeft de response van ESP32 terug.
        """
        cmd = cmd.strip().lower()

        if not self.enabled:
            print(f"[UART] (uitgeschakeld) commando: {cmd}")
            return f"OK: {cmd.upper()}"

        try:
            self.ser.write((cmd + '\n').encode())
            response = self.ser.readline().decode().strip()
            if response:
                return response
            else:
                return f"SENT: {cmd.upper()}"
        except Exception as exc:
            print(f"[UART] Fout bij verzenden '{cmd}': {exc}")
            return f"ERROR: {cmd}"

    def close(self) -> None:
        """Sluit de serial verbinding."""
        if self.ser:
            self.ser.close()


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    ctrl = LedController()
    for cmd in ["pmd", "rest", "karton", "organisch", "hit", "off"]:
        print(f"Verzonden: {cmd} -> {ctrl.send_command(cmd)}")
        time.sleep(2)
    ctrl.close()


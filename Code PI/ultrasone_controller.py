from __future__ import annotations

import threading
import time

try:
    import RPi.GPIO as GPIO  # type: ignore
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


class UltrasonicMonitor:
    """Background monitor for bin fill-level using HC-SR04 sensors."""

    SENSOR_MAP = {
        "rest": {"trig": 25, "echo": 26},
        "pmd": {"trig": 22, "echo": 4},
        "papier": {"trig": 16, "echo": 24},
        # Default Organisch sensor pins (can be overridden from CLI).
        "org": {"trig": 17, "echo": 27},
    }
    # Sensor points to the bin corner (diagonal line), so use calibrated
    # empty-bin reference distance instead of vertical height.
    EMPTY_DISTANCE_CM = 75.0
    THROW_THRESHOLD_CM = 10.0
    THROW_DEBOUNCE_S = 1.5
    THROW_CONFIRM_DELAY_S = 0.8
    THROW_CONFIRM_SAMPLES = 5
    THROW_STATUS_HOLD_S = 2.0

    def __init__(
        self,
        debug: bool = False,
        debug_every_cycles: int = 1,
        sensor_overrides: dict[str, dict[str, int] | None] | None = None,
    ) -> None:
        self.enabled = GPIO_AVAILABLE
        self.debug = bool(debug)
        self.debug_every_cycles = max(1, int(debug_every_cycles))
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._cycle_count = 0
        self.sensor_map = dict(self.SENSOR_MAP)
        if sensor_overrides:
            for key, cfg in sensor_overrides.items():
                if key in self.sensor_map:
                    self.sensor_map[key] = cfg
        self._status: dict[str, dict[str, object]] = {
            k: {
                "fill_pct": None,
                "is_full": False,
                "threw_object": False,
                "text": "onbekend" if self.sensor_map.get(k) else "n.v.t.",
            }
            for k in self.sensor_map
        }
        self._prev_distance: dict[str, float | None] = {k: None for k in self.sensor_map}
        self._last_throw_at: dict[str, float] = {k: 0.0 for k in self.sensor_map}
        if not self.enabled:
            return
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            for cfg in self.sensor_map.values():
                if not cfg:
                    continue
                GPIO.setup(cfg["trig"], GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(cfg["echo"], GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            time.sleep(0.2)
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self._debug_print("Ultrasonic monitor gestart")
        except Exception as exc:
            print(f"[Ultrasoon] Init mislukt: {exc}")
            self.enabled = False

    def close(self) -> None:
        if not self.enabled:
            return
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            GPIO.cleanup()
        except Exception:
            pass

    def snapshot(self) -> dict[str, dict[str, object]]:
        with self._lock:
            return {k: dict(v) for k, v in self._status.items()}

    def _loop(self) -> None:
        while self._running:
            self._cycle_count += 1
            should_print_cycle = self.debug and (self._cycle_count % self.debug_every_cycles == 0)
            for key, cfg in self.sensor_map.items():
                if not cfg:
                    continue
                d = self._measure_average(cfg, samples=3)
                fill = self._calc_fill_pct(d)
                is_full = (d is not None and (d < 5.0 or fill >= 95))
                now = time.time()
                prev = self._prev_distance.get(key)
                threw_object = self._detect_throw_event(key, cfg, prev, d, now)

                if d is not None:
                    self._prev_distance[key] = d

                if d is None:
                    txt = "geen meting"
                elif is_full:
                    txt = f"VOL ({fill}%)"
                elif threw_object or (now - self._last_throw_at.get(key, 0.0) <= self.THROW_STATUS_HOLD_S):
                    txt = f"inworp gedetecteerd ({fill}%)"
                elif fill >= 80:
                    txt = f"bijna vol ({fill}%)"
                else:
                    txt = f"{fill}%"
                if should_print_cycle:
                    trig = cfg["trig"]
                    echo = cfg["echo"]
                    d_txt = f"{d:.1f}cm" if d is not None else "None"
                    self._debug_print(
                        f"{key:<6} trig={trig:>2} echo={echo:>2} distance={d_txt:>8} fill={fill:>3} throw={threw_object} status={txt}"
                    )
                with self._lock:
                    self._status[key] = {
                        "fill_pct": fill if d is not None else None,
                        "is_full": is_full,
                        "threw_object": threw_object,
                        "text": txt,
                    }
                time.sleep(0.05)
            if should_print_cycle:
                self._debug_print("-" * 72)
            time.sleep(1.0)

    @staticmethod
    def _measure_distance(cfg: dict, timeout: float = 0.038) -> float | None:
        t0 = time.time()
        while GPIO.input(cfg["echo"]) == 1:
            if time.time() - t0 >= timeout:
                return None
        GPIO.output(cfg["trig"], GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(cfg["trig"], GPIO.LOW)

        t1 = time.time()
        while GPIO.input(cfg["echo"]) == 0:
            if time.time() - t1 >= timeout:
                return None
        start = time.time()
        while GPIO.input(cfg["echo"]) == 1:
            if time.time() - start >= timeout:
                return None
        stop = time.time()
        d = ((stop - start) * 34300.0) / 2.0
        return d if 2.0 < d <= 400.0 else None

    def _measure_average(self, cfg: dict, samples: int = 3) -> float | None:
        vals: list[float] = []
        for _ in range(samples):
            d = self._measure_distance(cfg)
            if d is not None:
                vals.append(d)
            time.sleep(0.02)
        return (sum(vals) / len(vals)) if vals else None

    def _debug_print(self, msg: str) -> None:
        if self.debug:
            print(f"[Ultrasoon DEBUG] {msg}")

    def _detect_throw_event(
        self,
        key: str,
        cfg: dict,
        prev_distance: float | None,
        current_distance: float | None,
        now: float,
    ) -> bool:
        """Detect a new throw-in event by measuring sudden, confirmed distance drop."""
        if prev_distance is None or current_distance is None:
            return False
        if now - self._last_throw_at.get(key, 0.0) < self.THROW_DEBOUNCE_S:
            return False
        if (prev_distance - current_distance) < self.THROW_THRESHOLD_CM:
            return False

        time.sleep(self.THROW_CONFIRM_DELAY_S)
        confirmed = self._measure_average(cfg, samples=self.THROW_CONFIRM_SAMPLES)
        if confirmed is None:
            return False
        if (prev_distance - confirmed) < self.THROW_THRESHOLD_CM:
            return False

        self._prev_distance[key] = confirmed
        self._last_throw_at[key] = now
        return True

    def _calc_fill_pct(self, distance_cm: float | None) -> int:
        if distance_cm is None:
            return -1
        pct = ((self.EMPTY_DISTANCE_CM - distance_cm) / self.EMPTY_DISTANCE_CM) * 100.0
        return max(0, min(100, int(pct)))

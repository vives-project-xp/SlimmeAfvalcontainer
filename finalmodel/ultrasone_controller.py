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
    THROW_THRESHOLD_CM = 15.0
    THROW_DEBOUNCE_S = 1.5
    THROW_CONFIRM_DELAY_S = 0.2
    THROW_CONFIRM_SAMPLES = 2
    THROW_STATUS_HOLD_S = 2.0
    REASON_LABELS = {
        "ok": "OK",
        "stuck_high_pretrigger": "ECHO_HIGH",
        "no_rising_echo": "NO_ECHO",
        "echo_too_long": "ECHO_LONG",
        "out_of_range": "RANGE",
        "unknown": "UNKNOWN",
    }

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
                "diag": "",
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
                d, diag = self._measure_average(cfg, samples=2)
                fill = self._calc_fill_pct(d)
                is_full = (d is not None and (d < 5.0 or fill >= 95))
                now = time.time()
                prev = self._prev_distance.get(key)
                threw_object = self._detect_throw_event(key, cfg, prev, d, now)

                if d is not None:
                    self._prev_distance[key] = d

                if d is None:
                    txt = ""
                elif is_full:
                    txt = f"VOL ({fill}%)"
                elif threw_object or (now - self._last_throw_at.get(key, 0.0) <= self.THROW_STATUS_HOLD_S):
                    txt = f"{fill}%"
                elif fill >= 80:
                    txt = f"bijna vol ({fill}%)"
                else:
                    txt = f"{fill}%"
                if should_print_cycle:
                    trig = cfg["trig"]
                    echo = cfg["echo"]
                    d_txt = f"{d:.1f}cm" if d is not None else "None"
                    reason = self.REASON_LABELS.get(str(diag.get("reason", "unknown")), str(diag.get("reason", "unknown")))
                    samples_txt = str(diag["samples"])
                    self._debug_print(
                        f"{key.upper():<6} | T:{trig:>2} E:{echo:>2} | D:{d_txt:>8} | F:{fill:>3}% | "
                        f"S:{diag['ok_samples']}/{diag['total_samples']} | R:{reason:<9} | "
                        f"THROW:{'Y' if threw_object else 'N'} | [{samples_txt}]"
                    )
                threw_visible = threw_object or (now - self._last_throw_at.get(key, 0.0) <= self.THROW_STATUS_HOLD_S)
                with self._lock:
                    self._status[key] = {
                        "fill_pct": fill if d is not None else None,
                        "is_full": is_full,
                        "threw_object": threw_visible,
                        "diag": str(diag.get("reason", "")),
                        "text": txt,
                    }
                time.sleep(0.01)
            if should_print_cycle:
                self._debug_print("-" * 72)
            time.sleep(0.05)

    @staticmethod
    def _measure_distance(cfg: dict, timeout: float = 0.038) -> tuple[float | None, str | None]:
        t0 = time.time()
        while GPIO.input(cfg["echo"]) == 1:
            if time.time() - t0 >= timeout:
                return None, "stuck_high_pretrigger"
        GPIO.output(cfg["trig"], GPIO.HIGH)
        time.sleep(0.00001)
        GPIO.output(cfg["trig"], GPIO.LOW)

        t1 = time.time()
        while GPIO.input(cfg["echo"]) == 0:
            if time.time() - t1 >= timeout:
                return None, "no_rising_echo"
        start = time.time()
        while GPIO.input(cfg["echo"]) == 1:
            if time.time() - start >= timeout:
                return None, "echo_too_long"
        stop = time.time()
        d = ((stop - start) * 34300.0) / 2.0
        if not (2.0 < d <= 400.0):
            return None, "out_of_range"
        return d, None

    def _measure_average(self, cfg: dict, samples: int = 3) -> tuple[float | None, dict[str, object]]:
        vals: list[float] = []
        sample_values: list[str] = []
        reason_counts: dict[str, int] = {}
        for _ in range(samples):
            d, reason = self._measure_distance(cfg)
            if d is not None:
                vals.append(d)
                sample_values.append(f"{d:.1f}")
            else:
                sample_values.append("None")
                reason_key = reason or "unknown"
                reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
            time.sleep(0.02)
        if vals:
            dominant_reason = "ok"
        elif reason_counts:
            dominant_reason = max(reason_counts.items(), key=lambda kv: kv[1])[0]
        else:
            dominant_reason = "unknown"
        diag: dict[str, object] = {
            "reason": dominant_reason,
            "ok_samples": len(vals),
            "total_samples": samples,
            "samples": ",".join(sample_values),
        }
        return ((sum(vals) / len(vals)) if vals else None), diag

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
        confirmed, _diag = self._measure_average(cfg, samples=self.THROW_CONFIRM_SAMPLES)
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

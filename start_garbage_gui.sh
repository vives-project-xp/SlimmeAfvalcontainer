#!/usr/bin/env bash
set -euo pipefail

# Gebruik 30s bootvenster om de camera actief te initialiseren/op te warmen.
/home/kobe/SlimmeAfvalcontainer/venv_ssp/bin/python - <<'PY' || true
import time
from picamera2 import Picamera2

warmup_seconds = 30.0
deadline = time.time() + warmup_seconds
attempt = 0
camera_ready = False

while time.time() < deadline:
    attempt += 1
    cam = None
    try:
        cam = Picamera2()
        cfg = cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        cam.configure(cfg)
        cam.start()
        time.sleep(1.0)
        cam.capture_array()
        camera_ready = True
        print(f"[boot] camera warmup success (attempt {attempt})")

        # Houd camera warm tijdens resterende warmup-venster.
        while time.time() < deadline:
            try:
                cam.capture_array()
            except Exception:
                break
            time.sleep(1.5)
        cam.stop()
        break
    except Exception as exc:
        print(f"[boot] camera warmup attempt {attempt} failed: {exc}")
        time.sleep(1.0)
    finally:
        if cam is not None:
            try:
                cam.stop()
            except Exception:
                pass

if not camera_ready:
    print(f"[boot] camera warmup incomplete after {warmup_seconds:.0f}s")
PY

cd '/home/kobe/SlimmeAfvalcontainer/Code PI'
exec /home/kobe/SlimmeAfvalcontainer/venv_ssp/bin/python '/home/kobe/SlimmeAfvalcontainer/Code PI/garbagedetection_gui.py'

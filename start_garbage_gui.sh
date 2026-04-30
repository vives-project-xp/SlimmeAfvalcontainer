#!/usr/bin/env bash
set -euo pipefail

cd '/home/kobe/SlimmeAfvalcontainer/Code PI'
exec /home/kobe/SlimmeAfvalcontainer/Code\ PI/.venv/bin/python '/home/kobe/SlimmeAfvalcontainer/Code PI/garbagedetection_gui.py' --fullscreen

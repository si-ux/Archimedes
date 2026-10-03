#!/usr/bin/env bash
# One-time Codespace setup: system libs MediaPipe needs + Python deps.
set -e
sudo apt-get update -q
sudo apt-get install -y -q libegl1 libgles2 libgl1
pip install -q -r requirements-dev.txt
pip install -q -e .
echo
echo "Ready. Start the demo with:  make demo   (then open the forwarded port 8000)"

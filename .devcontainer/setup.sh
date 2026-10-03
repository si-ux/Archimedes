#!/usr/bin/env bash
# One-time Codespace setup: system libs MediaPipe needs + Python deps.
# apt problems (e.g. a stale third-party repo key in the base image) must not
# stop the Python install, so the apt step is allowed to fail with a warning.
set -u

if ! sudo apt-get update -q; then
  echo "WARN: apt-get update reported errors (often a stale yarn repo key); continuing"
fi
if ! sudo apt-get install -y -q libegl1 libgles2 libgl1; then
  echo "WARN: could not install libEGL/libGL; the browser demo still works, the desktop MediaPipe path may not"
fi

set -e
python -m pip install -q -r requirements-dev.txt
python -m pip install -q -e .
echo
echo "Ready. Start the demo with:  make demo   (then open the forwarded port 8000)"

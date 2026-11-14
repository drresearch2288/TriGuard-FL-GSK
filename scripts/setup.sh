#!/usr/bin/env bash
# One-time setup on Linux / macOS / Git Bash (from the repository root). Needs Python 3.11 and git.
set -euo pipefail
python3.11 -m venv .venv
PY=.venv/bin/python; [ -x "$PY" ] || PY=.venv/Scripts/python.exe
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install -r requirements.txt
[ -d third_party/FLamby ] || git clone https://github.com/owkin/FLamby.git third_party/FLamby
git -C third_party/FLamby checkout edacf54d5211520583b0133d55ac39b6fda8324b
echo "Setup done. Next: see README.md, section 'Reproduce'."

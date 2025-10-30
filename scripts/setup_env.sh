#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "[setup_env] Conda not found. Falling back to Python venv." >&2
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  conda env create -f environment.yml
  conda activate tomato-ripeness
fi

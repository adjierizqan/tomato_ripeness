#!/usr/bin/env bash
set -euo pipefail

DATA=${1:-datasets/tomato/data.yaml}
DEVICE=${2:-0}

python src/val.py --weights runs/yolov12_s/tomato/weights/best.pt --data "$DATA" --device "$DEVICE"
python src/val.py --weights runs/yolov11_l/tomato/weights/best.pt --data "$DATA" --device "$DEVICE"
python src/val.py --weights runs/yolov12x_swinT/tomato/weights/best.pt --data "$DATA" --device "$DEVICE"

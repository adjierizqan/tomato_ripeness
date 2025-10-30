#!/usr/bin/env bash
set -euo pipefail

DATA=${1:-datasets/tomato/data.yaml}
DEVICE=${2:-0}

python src/train.py --cfg configs/yolov12_s.yaml --data "$DATA" --device "$DEVICE"
python src/train.py --cfg configs/yolov11_l.yaml --data "$DATA" --device "$DEVICE"
python src/train.py --cfg configs/yolov12x_swinT.yaml --data "$DATA" --device "$DEVICE"
python src/train.py --cfg configs/rf_detr_r50.yaml --data "$DATA" --device "$DEVICE"
python src/train.py --cfg configs/rf_detr_swinT.yaml --data "$DATA" --device "$DEVICE"

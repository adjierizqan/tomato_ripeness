# Tomato Ripeness Detection Benchmark Suite

This repository hosts an end-to-end benchmarking pipeline for three-class tomato
ripeness detection (`green`, `orange`, `red`). The project targets mAP50 ≥ 0.90
on the provided dataset using YOLOv12 (baseline and Swin-backed variants),
YOLOv11, RF-DETR, and an ensemble weighted boxes fusion (WBF) headliner.

## Repository Layout

```
tomato_ripeness/
├── configs/                 # Experiment YAMLs and augmentation profiles
├── datasets/tomato/         # Dataset stub (place images/labels here)
├── reports/                 # Benchmark report and generated figures
├── scripts/                 # Helper shell scripts (env setup, train/eval sweeps)
└── src/                     # Python source code (training, eval, inference)
```

Important configuration files:

- `configs/augmentations.yaml`: lite/strong augmentation presets.
- `configs/hyp_sweep.yaml`: hyperparameter sweep ranges for YOLOv12.
- `configs/*.yaml`: model-specific configuration files (YOLOv12, YOLOv12+Swin,
  YOLOv11, RF-DETR).
- `src/models/yolov12_swin/`: custom Swin backbone wrapper used by
  `configs/yolov12x_swinT.yaml`.

## Environment Setup

A Conda environment specification (`environment.yml`) and a pip-compatible
`requirements.txt` are provided. The helper script below automatically picks the
best available option:

```bash
bash scripts/setup_env.sh
```

If Conda is available this creates and activates `tomato-ripeness`; otherwise a
local Python virtual environment (`.venv`) is provisioned.

## Dataset Preparation

1. Download and unzip the tomato dataset so that the structure matches:
   - `datasets/tomato/images/{train,val,test}`
   - `datasets/tomato/labels/{train,val,test}`
   - Labels must be YOLO txt files with normalized coordinates.
2. Verify class names in `datasets/tomato/data.yaml` match the dataset order.
3. (Optional) Run dataset diagnostics:

```bash
python src/data/dataset_stats.py --data datasets/tomato/data.yaml --output reports/dataset_stats.json
```

To convert YOLO annotations to COCO JSON (for RF-DETR training/evaluation):

```bash
python src/data/yolo_to_coco.py --data datasets/tomato/data.yaml --split train --output datasets/tomato/annotations/train.json
```

## Training

### YOLOv12 / YOLOv11

```bash
python src/train.py --cfg configs/yolov12_s.yaml --data datasets/tomato/data.yaml --img 1280 --epochs 300 --device 0
python src/train.py --cfg configs/yolov11_l.yaml --data datasets/tomato/data.yaml --img 1280 --epochs 280 --device 0
```

### YOLOv12 + Swin Transformer Backbone

```bash
python src/train.py --cfg configs/yolov12x_swinT.yaml --data datasets/tomato/data.yaml --device 0
```

### RF-DETR (ResNet-50 / Swin-Tiny)

The training script dispatches to the `rfdetr` Python package. Provide the
custom config path produced from the template in `configs/` (update to your
local repo layout as needed):

```bash
python src/train.py --cfg configs/rf_detr_r50.yaml --data datasets/tomato/data.yaml --device 0
python src/train.py --cfg configs/rf_detr_swinT.yaml --data datasets/tomato/data.yaml --device 0
```

For rapid experiments use the bundled sweep grid:

```bash
python src/train.py --cfg configs/yolov12_s.yaml --data datasets/tomato/data.yaml --sweep configs/hyp_sweep.yaml --epochs 80 --device 0
```

The convenience script below runs every baseline sequentially (single GPU):

```bash
bash scripts/train_all.sh datasets/tomato/data.yaml 0
```

## Evaluation & Testing

```bash
python src/val.py --weights runs/yolov12_s/tomato/weights/best.pt --data datasets/tomato/data.yaml --split val
python src/test.py --weights runs/yolov12_s/tomato/weights/best.pt --data datasets/tomato/data.yaml
```

For RF-DETR supply its evaluation entry point and configuration:

```bash
python src/val.py --framework rf_detr --weights runs/rf_detr_swinT/best.pth --data datasets/tomato/data.yaml \
  --rf-config projects/rf_detr/configs/rf_detr_swinT_tomato.yaml --rf-entry-point rfdetr.val
```

Batch evaluation for the YOLO checkpoints:

```bash
bash scripts/eval_all.sh datasets/tomato/data.yaml 0
```

## Inference & Export

Single-image / directory prediction:

```bash
python src/infer.py --weights runs/yolov12x_swinT/tomato/weights/best.pt --source datasets/tomato/images/test --conf 0.25
```

Model export (ONNX/TorchScript/TFLite supported by Ultralytics):

```bash
python src/export.py --weights runs/yolov12x_swinT/tomato/weights/best.pt --format onnx
```

## Ensemble Weighted Boxes Fusion

Fuse the top-performing YOLO checkpoints with optional evaluation:

```bash
python src/ensemble_wbf.py \
  --weights runs/yolov12x_swinT/tomato/weights/best.pt runs/yolov11_l/tomato/weights/best.pt \
  --data datasets/tomato/data.yaml --split test --img 1280 --weights-wbf 1.2 1.0 --eval
```

The script writes YOLO-format ensemble predictions to
`runs/ensemble/predictions/<split>/` and (when `--eval` is set) reports COCO
metrics using pycocotools.

## Reporting

Populate `reports/benchmark.md` with experiment outcomes. Include:

- Model configuration (parameters, GFLOPs, inference speed).
- Validation/test metrics (mAP50, mAP50-95, per-class AP, precision, recall).
- Latency/FPS and VRAM usage snapshots.
- Confusion matrices and PR curves (store in `reports/figures/`).

Use TensorBoard or W&B for monitoring by enabling the logger in the training
scripts.

## Reproducibility Checklist

- Use seed `42` and enable deterministic behaviour (`torch.use_deterministic_algorithms(True)`).
- Save a copy of the configuration YAML and git commit hash alongside each run
  in the `runs/` directory (Ultralytics handles this automatically).
- Document hardware specs (GPU model, CUDA driver, RAM) in `reports/benchmark.md`.

## Next Steps

- Explore higher input resolutions (e.g. 1536) and/or test-time augmentation for
  the orange class, which is typically the hardest to disambiguate.
- Investigate tiling or sliding-window inference if small tomato instances are
  missed at 1280 resolution.
- Calibrate ensemble weights and confidence thresholds to squeeze out the last
  points toward the 0.95 mAP50 goal.

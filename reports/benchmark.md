# Tomato Ripeness Detection Benchmark

> Replace the placeholder metrics below with actual experiment results once
> training/evaluation is complete.

## Summary Table

| Model | Params (M) | GFLOPs | Size (MB) | mAP50 | mAP50-95 | Precision | Recall | Latency (ms/img) | Throughput (FPS) | VRAM (GB) | Notes |
|-------|------------|--------|-----------|-------|----------|-----------|--------|------------------|------------------|-----------|-------|
| YOLOv12-s | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Baseline 1280px |
| YOLOv12x + Swin-T | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Swin backbone, freeze 15 epochs |
| YOLOv11-L | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| RF-DETR R50 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| RF-DETR Swin-T | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| Ensemble (WBF) | — | — | — | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Weighted boxes fusion |

## Per-Class AP (mAP50)

| Class | YOLOv12-s | YOLOv12x Swin-T | YOLOv11-L | RF-DETR R50 | RF-DETR Swin-T | Ensemble |
|-------|-----------|-----------------|-----------|-------------|----------------|----------|
| green | TBD | TBD | TBD | TBD | TBD | TBD |
| orange | TBD | TBD | TBD | TBD | TBD | TBD |
| red | TBD | TBD | TBD | TBD | TBD | TBD |

## Precision-Recall Curves

Place PR curve PNGs generated via `src/utils/plots.py` in `reports/figures/` and
reference them here, e.g.:

![PR Curve YOLOv12x Swin-T](figures/pr_yolov12x_swinT.png)

## Confusion Matrices

Document confusion matrices for the best-performing checkpoints.

## Latency & Throughput

Record inference benchmarks using representative hardware (RTX 4090, CUDA 12.x).
Include both batch=1 latency and steady-state FPS with larger batches.

## Observations & Recommendations

- Highlight the weakest class (likely `orange`) and discuss targeted
  augmentations or loss weighting to improve recall.
- Document stability improvements from Swin backbone freezing/warmup.
- Capture any training instabilities (NaNs, diverging losses) encountered during
  long runs.

## Next Experiments

- Try RT-DETR as an additional baseline if time permits.
- Explore TTA (scale/flip) during inference.
- Evaluate tiling for small-object recall.

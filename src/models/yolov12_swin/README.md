# YOLOv12 + Swin Transformer Backbone

This module provides a Swin Transformer backbone wrapper that is compatible with
Ultralytics' YOLOv12 detection head. The implementation relies on
[`timm`](https://github.com/huggingface/pytorch-image-models) to load a
Swin-Tiny/Small/Base backbone that emits three feature pyramid stages with
stride {8, 16, 32}. The helper class exposes a `forward` method returning the
list of feature maps expected by the YOLO neck.

Key design notes:

- Patch embed stem keeps the original 4× patch size, therefore images should be
  resized to multiples of 32 (1280 is the recommended default).
- The wrapper supports checkpoint loading from ImageNet-1k pretrained weights
  and exposes a utility to freeze the backbone for the warm-up period.
- Neck and head configuration live in `yolo_swin_cfg.yaml`, which extends the
  Ultralytics YAML schema with custom backbone import (`backbone_module`) and
  an optional `freeze_backbone_epochs` field consumed by `src/train.py`.

See `configs/yolov12x_swinT.yaml` for a concrete experiment configuration.

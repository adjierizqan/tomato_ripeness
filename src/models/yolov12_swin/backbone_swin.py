"""Swin Transformer backbone wrapper for YOLOv12 detection models.

The Ultralytics trainer expects a backbone that returns a list of feature maps
for strides {8, 16, 32}. This module adapts a timm Swin Transformer to that
interface and optionally loads ImageNet checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from torch import nn

try:
    import timm
except ImportError as exc:  # pragma: no cover - handled in setup instructions
    raise ImportError(
        "timm is required for Swin backbone support. Install with `pip install timm`."
    ) from exc


@dataclass
class SwinBackboneConfig:
    """Configuration for the Swin backbone wrapper."""

    model_name: str = "swin_tiny_patch4_window7_224"
    out_indices: Iterable[int] = (1, 2, 3)
    pretrained_path: Optional[str] = None
    freeze_at_start: bool = True


class SwinBackbone(nn.Module):
    """Wrap a Swin Transformer backbone to emit feature pyramid tensors."""

    def __init__(self, cfg: SwinBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(
            cfg.model_name,
            pretrained=cfg.pretrained_path is None,
            features_only=True,
            out_indices=tuple(cfg.out_indices),
        )

        if cfg.pretrained_path:
            checkpoint = torch.load(Path(cfg.pretrained_path), map_location="cpu")
            missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
            if missing or unexpected:
                print(
                    f"[SwinBackbone] Loaded with missing={missing}, unexpected={unexpected}"
                )

        if cfg.freeze_at_start:
            self.freeze_stages()

    def freeze_stages(self, stages: Optional[Iterable[int]] = None) -> None:
        """Freeze parameter gradients for the selected stages."""

        if stages is None:
            stages = range(len(self.model.feature_info))
        for idx, module in enumerate(self.model.blocks):  # type: ignore[attr-defined]
            requires_grad = idx not in stages
            for param in module.parameters():
                param.requires_grad = requires_grad
        # Freeze patch embedding as well when requested
        if 0 in stages:
            for param in self.model.patch_embed.parameters():
                param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:  # type: ignore[override]
        features = self.model(x)
        if not isinstance(features, list):
            raise TypeError("Expected a list of feature maps from Swin backbone")
        return features


def build_swin_backbone(cfg_dict: Optional[dict] = None) -> SwinBackbone:
    """Factory that instantiates :class:`SwinBackbone` from a dictionary."""

    cfg = SwinBackboneConfig(**(cfg_dict or {}))
    return SwinBackbone(cfg)

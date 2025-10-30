"""Lightweight W&B logger wrapper (optional)."""
from __future__ import annotations

from typing import Any, Dict, Optional


class WandbLogger:
    def __init__(self, project: str, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install wandb to enable logging: pip install wandb") from exc
        self._wandb = wandb
        self.run = wandb.init(project=project, name=name, config=config)

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        self._wandb.log(data, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

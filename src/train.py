"""Unified training entry point for YOLOv12/YOLOv11 and RF-DETR models."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def resolve_augmentations(tag: str | None) -> Dict[str, Any]:
    if not tag:
        return {}
    if "::" not in tag:
        raise ValueError("Augmentation tag must follow '<path>::<profile>' format")
    path_str, profile = tag.split("::", 1)
    cfg = load_yaml(Path(path_str))
    try:
        return cfg["profiles"][profile]
    except KeyError as exc:
        raise KeyError(f"Augmentation profile '{profile}' not found in {path_str}") from exc


def train_ultralytics(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    from ultralytics import YOLO  # type: ignore

    model = YOLO(cfg["model"])
    train_args: Dict[str, Any] = {
        "data": str(args.data),
        "imgsz": args.img or cfg.get("imgsz", 1280),
        "epochs": args.epochs or cfg.get("epochs", 300),
        "batch": args.batch or cfg.get("batch", "auto"),
        "project": cfg.get("project"),
        "name": cfg.get("name"),
        "patience": cfg.get("patience", 30),
        "optimizer": cfg.get("optimizer", "AdamW"),
        "lr0": cfg.get("lr0"),
        "lrf": cfg.get("lrf"),
        "warmup_epochs": cfg.get("warmup_epochs", 3),
        "weight_decay": cfg.get("weight_decay", 0.0005),
        "seed": cfg.get("seed", 42),
        "deterministic": cfg.get("deterministic", True),
        "val": True,
        "save_json": cfg.get("save_json", True),
    }

    augmentations = resolve_augmentations(cfg.get("augmentations"))
    train_args.update({k: v for k, v in augmentations.items() if k not in {"img_size", "multi_scale"}})

    if cfg.get("notes"):
        print("\n=== Experiment Notes ===")
        print(cfg["notes"])
        print("========================\n")

    if args.sweep:
        sweep_cfg = load_yaml(Path(args.sweep))
        train_args["cfg"] = sweep_cfg
    if args.device:
        train_args["device"] = args.device

    results = model.train(**{k: v for k, v in train_args.items() if v is not None})
    if results:
        print(json.dumps(results, indent=2, default=str))


def train_rf_detr(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    config_path = cfg.get("rf_detr_config")
    if not config_path:
        raise ValueError("RF-DETR config must define 'rf_detr_config'")
    command = [
        "python",
        "-m",
        cfg.get("entry_point", "rfdetr.train"),
        "--config",
        config_path,
        "--output",
        str(Path(cfg.get("project", "runs")) / cfg.get("name", "rf_detr")),
        "--imgsz",
        str(args.img or cfg.get("imgsz", 1280)),
        "--epochs",
        str(args.epochs or cfg.get("epochs", 300)),
    ]
    if args.device:
        command.extend(["--device", args.device])
    print("Executing RF-DETR command:", " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=Path, required=True, help="Experiment YAML configuration")
    parser.add_argument("--data", type=Path, required=True, help="Dataset data.yaml path")
    parser.add_argument("--img", type=int, default=None, help="Override image size")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch", default=None, help="Override batch size")
    parser.add_argument("--device", default=None, help="CUDA device id")
    parser.add_argument("--sweep", type=Path, default=None, help="Optional hyperparameter sweep config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.cfg)
    framework = cfg.get("framework", "ultralytics")

    if framework == "ultralytics":
        train_ultralytics(cfg, args)
    elif framework == "rf_detr":
        train_rf_detr(cfg, args)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


if __name__ == "__main__":
    main()

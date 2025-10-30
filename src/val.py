"""Validation script for trained detectors."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict


def validate_ultralytics(weights: Path, args: argparse.Namespace) -> Dict[str, Any]:
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(weights))
    results = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.img or 1280,
        device=args.device,
        save_json=True,
        project=args.project,
        name=args.name,
        batch=args.batch,
    )
    summary = {
        "map50": getattr(results, "box", {}).get("map50", None),
        "map5095": getattr(results, "box", {}).get("map", None),
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary


def validate_rf_detr(weights: Path, args: argparse.Namespace) -> Dict[str, Any]:
    command = [
        "python",
        "-m",
        args.rf_entry_point,
        "--config",
        args.rf_config,
        "--weights",
        str(weights),
        "--split",
        args.split,
        "--output",
        str(Path(args.project) / args.name),
    ]
    if args.device:
        command.extend(["--device", args.device])
    print("Executing RF-DETR validation:", " ".join(command))
    subprocess.run(command, check=True)
    return {"status": "completed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--img", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", default=None)
    parser.add_argument("--project", default="runs/eval")
    parser.add_argument("--name", default="eval")
    parser.add_argument("--framework", choices=["ultralytics", "rf_detr"], default="ultralytics")
    parser.add_argument("--rf-config", dest="rf_config", type=str, default="")
    parser.add_argument("--rf-entry-point", dest="rf_entry_point", type=str, default="rfdetr.val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.framework == "ultralytics":
        validate_ultralytics(args.weights, args)
    else:
        if not args.rf_config:
            raise ValueError("--rf-config is required for RF-DETR evaluation")
        validate_rf_detr(args.weights, args)


if __name__ == "__main__":
    main()

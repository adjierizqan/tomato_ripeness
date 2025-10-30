"""Model export helper supporting ONNX/TorchScript for YOLO models."""
from __future__ import annotations

import argparse
from pathlib import Path


def export_ultralytics(weights: Path, format: str, imgsz: int, device: str | None) -> None:
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(weights))
    model.export(format=format, imgsz=imgsz, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--format", type=str, default="onnx")
    parser.add_argument("--img", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--framework", choices=["ultralytics"], default="ultralytics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.framework == "ultralytics":
        export_ultralytics(args.weights, args.format, args.img, args.device)
    else:  # pragma: no cover - we only support YOLO export currently
        raise ValueError("Export only implemented for Ultralytics models")


if __name__ == "__main__":
    main()

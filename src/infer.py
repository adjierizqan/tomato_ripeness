"""Inference helper for YOLO and RF-DETR models."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def infer_ultralytics(weights: Path, source: Path | str, imgsz: int, device: str | None, conf: float) -> None:
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        imgsz=imgsz,
        device=device,
        conf=conf,
        save=True,
        save_txt=True,
        save_conf=True,
    )


def infer_rf_detr(weights: Path, source: Path | str, imgsz: int, device: str | None, conf: float) -> None:
    import subprocess

    command: List[str] = [
        "python",
        "-m",
        "rfdetr.infer",
        "--weights",
        str(weights),
        "--source",
        str(source),
        "--imgsz",
        str(imgsz),
        "--conf-thres",
        str(conf),
    ]
    if device:
        command.extend(["--device", device])
    print("Executing RF-DETR inference:", " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=str, default="datasets/tomato/images/test")
    parser.add_argument("--img", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--framework", choices=["ultralytics", "rf_detr"], default="ultralytics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.framework == "ultralytics":
        infer_ultralytics(args.weights, args.source, args.img, args.device, args.conf)
    else:
        infer_rf_detr(args.weights, args.source, args.img, args.device, args.conf)


if __name__ == "__main__":
    main()

"""Test script wrapper that mirrors validation but defaults to the test split."""
from __future__ import annotations

import argparse
from pathlib import Path

from val import main as val_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--img", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", default=None)
    parser.add_argument("--project", default="runs/test")
    parser.add_argument("--name", default="test")
    parser.add_argument("--framework", choices=["ultralytics", "rf_detr"], default="ultralytics")
    parser.add_argument("--rf-config", dest="rf_config", type=str, default="")
    parser.add_argument("--rf-entry-point", dest="rf_entry_point", type=str, default="rfdetr.test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import sys

    sys.argv = [
        "val",
        "--weights",
        str(args.weights),
        "--data",
        str(args.data),
        "--split",
        "test",
        "--project",
        args.project,
        "--name",
        args.name,
        "--framework",
        args.framework,
    ]
    if args.img is not None:
        sys.argv.extend(["--img", str(args.img)])
    if args.device is not None:
        sys.argv.extend(["--device", str(args.device)])
    if args.batch is not None:
        sys.argv.extend(["--batch", str(args.batch)])
    if args.framework == "rf_detr":
        if not args.rf_config:
            raise ValueError("--rf-config required for RF-DETR testing")
        sys.argv.extend(["--rf-config", args.rf_config, "--rf-entry-point", args.rf_entry_point])
    val_main()


if __name__ == "__main__":
    main()

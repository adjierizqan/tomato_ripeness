"""Utility to compute dataset statistics for the tomato ripeness dataset."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from PIL import Image, ImageOps


def load_yaml(path: Path) -> Dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def read_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image


def scan_split(image_dir: Path) -> Tuple[List[Tuple[int, int]], Counter]:
    shapes: List[Tuple[int, int]] = []
    aspect_ratios: Counter = Counter()
    for image_path in image_dir.glob("*"):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = read_image(image_path)
        width, height = image.size
        shapes.append((width, height))
        ratio_bucket = round(width / height, 2)
        aspect_ratios[ratio_bucket] += 1
    return shapes, aspect_ratios


def summarize_shapes(shapes: List[Tuple[int, int]]) -> Dict[str, float]:
    if not shapes:
        return {"count": 0}
    widths, heights = zip(*shapes)
    return {
        "count": len(shapes),
        "width_mean": sum(widths) / len(widths),
        "height_mean": sum(heights) / len(heights),
        "width_min": min(widths),
        "width_max": max(widths),
        "height_min": min(heights),
        "height_max": max(heights),
    }


def compute_label_distribution(labels_dir: Path, class_names: List[str]) -> Dict[str, int]:
    counter = Counter()
    for label_path in labels_dir.glob("*.txt"):
        with label_path.open("r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                class_id = int(line.split()[0])
                class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
                counter[class_name] += 1
    return dict(counter)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    data_cfg = load_yaml(args.data)
    class_names: List[str] = data_cfg.get("names") or data_cfg.get("names", [])

    report: Dict[str, Dict] = {}
    for split in ("train", "val", "test"):
        image_dir = Path(data_cfg[split] if split in data_cfg else data_cfg["path"]) / "images" / split
        label_dir = Path(data_cfg[split] if split in data_cfg else data_cfg["path"]) / "labels" / split
        shapes, ratios = scan_split(image_dir)
        report[split] = {
            "shape_stats": summarize_shapes(shapes),
            "aspect_ratio_histogram": ratios,
            "label_distribution": compute_label_distribution(label_dir, class_names),
        }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(report, handle, indent=2)
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

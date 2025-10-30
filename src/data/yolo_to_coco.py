"""Convert YOLO txt annotations to COCO JSON format."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml


def load_yaml(path: Path) -> Dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output", type=Path, required=True, help="Output COCO JSON path")
    return parser.parse_args()


def load_labels(label_dir: Path) -> List[Dict]:
    annotations: List[Dict] = []
    ann_id = 1
    for img_id, label_path in enumerate(sorted(label_dir.glob("*.txt")), start=1):
        with label_path.open("r") as handle:
            for line in handle:
                if not line.strip():
                    continue
                class_id, x_center, y_center, width, height = map(float, line.split())
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": int(class_id) + 1,
                        "bbox": [x_center, y_center, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
    return annotations


def build_images(image_dir: Path) -> List[Dict]:
    images: List[Dict] = []
    for img_id, image_path in enumerate(sorted(image_dir.glob("*")), start=1):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        images.append({"id": img_id, "file_name": image_path.name})
    return images


def main() -> None:
    args = parse_args()
    data_cfg = load_yaml(args.data)
    base_path = Path(data_cfg.get("path", args.data.parent))
    image_dir = base_path / "images" / args.split
    label_dir = base_path / "labels" / args.split

    categories = [
        {"id": idx + 1, "name": name, "supercategory": "tomato"}
        for idx, name in enumerate(data_cfg["names"])
    ]

    coco_dict = {
        "images": build_images(image_dir),
        "annotations": load_labels(label_dir),
        "categories": categories,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(coco_dict, handle)
    print(f"Saved COCO annotations to {args.output}")


if __name__ == "__main__":
    main()

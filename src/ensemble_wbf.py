"""Weighted boxes fusion ensemble for tomato ripeness detectors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

from data.yolo_to_coco import load_yaml as load_data_yaml


BoxList = List[List[float]]
ScoreList = List[float]
LabelList = List[int]


def list_dataset_images(data_yaml: Path, split: str) -> List[Path]:
    data_cfg = load_data_yaml(data_yaml)
    base_path = Path(data_cfg.get("path", data_yaml.parent))
    image_dir = base_path / "images" / split
    return sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )


def predict_ultralytics(
    weights: Path,
    images: List[Path],
    imgsz: int,
    device: str | None,
    conf: float,
) -> Tuple[Dict[str, BoxList], Dict[str, ScoreList], Dict[str, LabelList], Dict[str, Tuple[int, int]]]:
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(weights))
    boxes: Dict[str, BoxList] = {}
    scores: Dict[str, ScoreList] = {}
    labels: Dict[str, LabelList] = {}
    shapes: Dict[str, Tuple[int, int]] = {}

    for result in model.predict(
        source=[str(p) for p in images],
        imgsz=imgsz,
        device=device,
        conf=conf,
        stream=True,
        save=False,
    ):
        path = Path(result.path).name
        w, h = result.orig_shape[1], result.orig_shape[0]
        shapes[path] = (w, h)
        if result.boxes is None or len(result.boxes) == 0:
            boxes[path] = []
            scores[path] = []
            labels[path] = []
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        boxes[path] = [[x1 / w, y1 / h, x2 / w, y2 / h] for x1, y1, x2, y2 in xyxy]
        scores[path] = confs.tolist()
        labels[path] = cls.tolist()
    return boxes, scores, labels, shapes


def run_wbf(
    per_model_boxes: List[Dict[str, BoxList]],
    per_model_scores: List[Dict[str, ScoreList]],
    per_model_labels: List[Dict[str, LabelList]],
    image_shapes: Dict[str, Tuple[int, int]],
    weights: Sequence[float],
    iou_thr: float,
    skip_box_thr: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    fused: Dict[str, Dict[str, np.ndarray]] = {}
    image_names = image_shapes.keys()
    for name in image_names:
        boxes_list = [model_boxes.get(name, []) for model_boxes in per_model_boxes]
        scores_list = [model_scores.get(name, []) for model_scores in per_model_scores]
        labels_list = [model_labels.get(name, []) for model_labels in per_model_labels]
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        fused[name] = {
            "boxes": np.array(fused_boxes),
            "scores": np.array(fused_scores),
            "labels": np.array(fused_labels, dtype=int),
        }
    return fused


def save_yolo_predictions(
    fused: Dict[str, Dict[str, np.ndarray]],
    shapes: Dict[str, Tuple[int, int]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in fused.items():
        w, h = shapes[name]
        boxes = payload["boxes"]
        scores = payload["scores"]
        labels = payload["labels"]
        txt_path = output_dir / f"{Path(name).stem}.txt"
        with txt_path.open("w") as handle:
            for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
                xc = (x1 + x2) / 2
                yc = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1
                handle.write(f"{label} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {score:.4f}\n")


def coco_evaluate(
    data_yaml: Path,
    split: str,
    fused: Dict[str, Dict[str, np.ndarray]],
    shapes: Dict[str, Tuple[int, int]],
    output_json: Path,
) -> None:
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore

    data_cfg = load_data_yaml(data_yaml)
    base_path = Path(data_cfg.get("path", data_yaml.parent))
    label_dir = base_path / "labels" / split
    image_dir = base_path / "images" / split

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    categories = [
        {"id": idx + 1, "name": name, "supercategory": "tomato"}
        for idx, name in enumerate(data_cfg["names"])
    ]

    for img_id, label_path in enumerate(sorted(label_dir.glob("*.txt")), start=1):
        stem = label_path.stem
        candidates = [name for name in shapes.keys() if Path(name).stem == stem]
        if candidates:
            image_name = candidates[0]
            width, height = shapes[image_name]
        else:
            image_path = next(image_dir.glob(f"{stem}.*"), None)
            if image_path and image_path.exists():
                from PIL import Image  # local import to avoid mandatory dependency at module import time

                width, height = Image.open(image_path).size
                image_name = image_path.name
            else:
                image_name = f"{stem}.jpg"
                width, height = (1, 1)
        shapes[image_name] = (width, height)
        images.append({"id": img_id, "file_name": image_name, "width": width, "height": height})
        with label_path.open("r") as handle:
            for ann_id, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                class_id, xc, yc, bw, bh = map(float, line.split()[:5])
                x = (xc - bw / 2) * width
                y = (yc - bh / 2) * height
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": img_id,
                        "category_id": int(class_id) + 1,
                        "bbox": [x, y, bw * width, bh * height],
                        "iscrowd": 0,
                        "area": bw * bh * width * height,
                    }
                )

    detections: List[Dict[str, Any]] = []
    name_to_id = {img["file_name"]: img["id"] for img in images}
    for name, payload in fused.items():
        width, height = shapes[name]
        image_id = name_to_id.get(name)
        if image_id is None:
            continue
        for (x1, y1, x2, y2), score, label in zip(
            payload["boxes"], payload["scores"], payload["labels"]
        ):
            x = x1 * width
            y = y1 * height
            w = (x2 - x1) * width
            h = (y2 - y1) * height
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [x, y, w, h],
                    "score": float(score),
                }
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(detections, handle)

    coco_gt = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    gt_json_path = output_json.with_name("gt_tmp.json")
    with gt_json_path.open("w") as handle:
        json.dump(coco_gt, handle)

    cocoGt = COCO(str(gt_json_path))
    cocoDt = cocoGt.loadRes(str(output_json))
    coco_eval = COCOeval(cocoGt, cocoDt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--frameworks", nargs="+", default=None)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--img", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou-thr", type=float, default=0.6)
    parser.add_argument("--skip-box-thr", type=float, default=0.001)
    parser.add_argument("--weights-wbf", nargs="+", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/ensemble/predictions"))
    parser.add_argument("--eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frameworks = args.frameworks or ["ultralytics"] * len(args.weights)
    if len(frameworks) != len(args.weights):
        raise ValueError("--frameworks must match the number of weights")

    images = list_dataset_images(args.data, args.split)
    per_model_boxes: List[Dict[str, BoxList]] = []
    per_model_scores: List[Dict[str, ScoreList]] = []
    per_model_labels: List[Dict[str, LabelList]] = []
    shapes: Dict[str, Tuple[int, int]] = {}

    for weight, framework in zip(args.weights, frameworks):
        if framework != "ultralytics":
            raise NotImplementedError(
                "RF-DETR ensemble inference requires exporting predictions to YOLO format first."
            )
        boxes, scores, labels, local_shapes = predict_ultralytics(weight, images, args.img, args.device, args.conf)
        shapes.update(local_shapes)
        per_model_boxes.append(boxes)
        per_model_scores.append(scores)
        per_model_labels.append(labels)

    weights_wbf = args.weights_wbf or [1.0] * len(per_model_boxes)
    fused = run_wbf(
        per_model_boxes,
        per_model_scores,
        per_model_labels,
        shapes,
        weights_wbf,
        args.iou_thr,
        args.skip_box_thr,
    )

    save_yolo_predictions(fused, shapes, args.output / args.split)

    if args.eval:
        coco_evaluate(
            args.data,
            args.split,
            fused,
            shapes,
            args.output / f"ensemble_{args.split}.json",
        )


if __name__ == "__main__":
    main()

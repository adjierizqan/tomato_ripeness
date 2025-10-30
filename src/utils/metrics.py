"""Helpers for collecting detection metrics across frameworks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DetectionMetrics:
    """Container for AP and AR metrics compatible with YOLO/RF-DETR outputs."""

    map50: float = 0.0
    map5095: float = 0.0
    per_class_ap: Dict[str, float] = field(default_factory=dict)
    precision: float = 0.0
    recall: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        payload = {
            "map50": self.map50,
            "map5095": self.map5095,
            "precision": self.precision,
            "recall": self.recall,
        }
        payload.update({f"ap/{cls}": value for cls, value in self.per_class_ap.items()})
        return payload


def merge_metrics(metrics_list: List[DetectionMetrics]) -> DetectionMetrics:
    if not metrics_list:
        return DetectionMetrics()
    merged = DetectionMetrics()
    merged.map50 = sum(m.map50 for m in metrics_list) / len(metrics_list)
    merged.map5095 = sum(m.map5095 for m in metrics_list) / len(metrics_list)
    merged.precision = sum(m.precision for m in metrics_list) / len(metrics_list)
    merged.recall = sum(m.recall for m in metrics_list) / len(metrics_list)
    classes = set().union(*(m.per_class_ap.keys() for m in metrics_list))
    merged.per_class_ap = {
        cls: sum(m.per_class_ap.get(cls, 0.0) for m in metrics_list) / len(metrics_list)
        for cls in classes
    }
    return merged

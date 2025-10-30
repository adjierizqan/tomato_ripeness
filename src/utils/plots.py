"""Plotting utilities for detection experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def save_pr_curve(pr_dict: Dict[str, Dict[str, Iterable[float]]], output_path: Path) -> None:
    """Save precision-recall curves per class.

    Args:
        pr_dict: mapping class_name -> {"precision": [...], "recall": [...]} values.
    """

    plt.figure(figsize=(6, 6))
    for cls, values in pr_dict.items():
        precision = list(values.get("precision", []))
        recall = list(values.get("recall", []))
        if not precision or not recall:
            continue
        plt.plot(recall, precision, label=cls)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm, class_names: List[str], output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

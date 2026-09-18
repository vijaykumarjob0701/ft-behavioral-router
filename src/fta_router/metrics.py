"""Routing evaluation metrics: accuracy, macro-F1, confusion matrix."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from fta_router.schema import PRIMARY_ACTIONS


def routing_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
) -> dict:
    """Compute accuracy, macro-F1, and confusion matrix for primary_action."""
    labels = list(labels) if labels is not None else list(PRIMARY_ACTIONS)
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def format_confusion(cm: list[list[int]], labels: Sequence[str]) -> str:
    header = "pred→\\true↓ | " + " | ".join(f"{l:>14}" for l in labels)
    lines = [header, "-" * len(header)]
    for i, lab in enumerate(labels):
        row = " | ".join(f"{cm[i][j]:14d}" for j in range(len(labels)))
        lines.append(f"{lab:>14} | {row}")
    return "\n".join(lines)

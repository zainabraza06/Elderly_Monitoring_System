from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro")),
    }

    uniq = np.unique(y_true)
    if len(uniq) == 2:
        out["sensitivity"] = float(recall_score(y_true, y_pred, pos_label=1))
        out["specificity"] = float(recall_score(y_true, y_pred, pos_label=0))
        if y_prob is not None:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_file: Path, title: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_file: Path, title: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def save_metrics_report(metrics: Dict[str, Dict[str, float]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, vals in metrics.items():
        row = {"task": task}
        row.update(vals)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_file, index=False)

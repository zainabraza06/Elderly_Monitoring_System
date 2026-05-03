"""Save ADL training figures under results/ (no display in headless CI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def save_model_comparison_bar(
    results: list[dict[str, Any]],
    out_path: Path,
    *,
    title: str = "ADL classification — model comparison",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [r["Model"] for r in results]
    acc = [r["Accuracy"] for r in results]
    f1 = [r["F1-Score"] for r in results]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, acc, w, label="Accuracy", color="#2E86AB")
    ax.bar(x + w / 2, f1, w, label="Weighted F1", color="#F18F01")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_png(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    out_path: Path,
    *,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm / (cm.sum(axis=1, keepdims=True) + 1e-9), cmap="Blues", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.colorbar(im, ax=ax)
    tick = np.arange(len(class_labels))
    ax.set_xticks(tick)
    ax.set_yticks(tick)
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

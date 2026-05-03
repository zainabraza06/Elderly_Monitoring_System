"""Figures for fall-binary training reports. Matplotlib loaded lazily."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_confusion_matrix_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    *,
    labels: tuple[str, str] = ("Non-fall", "Fall"),
    title: str,
) -> None:
    plt = _pyplot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    try:
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn", ax=ax, xticklabels=labels, yticklabels=labels)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    except ImportError:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="RdYlGn")
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_bars(
    results: dict[str, dict[str, float]],
    out_path: Path,
    *,
    title: str = "Fall detection — model comparison",
    f1_label: str = "F1 (fall class)",
) -> None:
    plt = _pyplot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    acc = [results[m]["accuracy"] * 100 for m in names]
    f1 = [results[m]["f1"] * 100 for m in names]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, acc, w, label="Accuracy", color="#4CAF50")
    ax.bar(x + w / 2, f1, w, label=f1_label, color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_binary_per_class_bars(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    labels: list[str],
    out_path: Path,
    *,
    title: str = "Per-class performance (fall detection)",
) -> None:
    """Two-class precision/recall/F1 bars."""
    plt = _pyplot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, precision, w, label="Precision", color="#66BB6A")
    ax.bar(x, recall, w, label="Recall", color="#42A5F5")
    ax.bar(x + w, f1, w, label="F1", color="#FF7043")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

"""Figures for fall-type training reports (Colab STEP 9 style). Matplotlib loaded lazily."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    out_path: Path,
    *,
    title: str,
) -> None:
    plt = _pyplot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    try:
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="RdYlGn",
            ax=ax,
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    except ImportError:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap="RdYlGn")
        ax.set_title(title)
        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticklabels(class_labels)
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_model_comparison_bars(
    results: dict[str, dict[str, float]],
    out_path: Path,
    *,
    title: str = "Fall-type model comparison",
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
    ax.bar(x + w / 2, f1, w, label="Weighted F1", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_per_class_bars(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    class_labels: list[str],
    out_path: Path,
    *,
    title: str = "Per-class performance",
) -> None:
    plt = _pyplot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(class_labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, precision, w, label="Precision", color="#66BB6A")
    ax.bar(x, recall, w, label="Recall", color="#42A5F5")
    ax.bar(x + w, f1, w, label="F1", color="#FF7043")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_sensor_contribution_pie(out_path: Path) -> None:
    """Illustrative pie from Colab (feature group counts — approximate)."""
    plt = _pyplot()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sizes = [180, 84, 15, 50, 15, 1]
    labels = ["Accelerometer", "Gyroscope", "Orientation", "Fall-specific", "Cross-sensor", "SMA"]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, explode=[0.02] * 6)
    ax.set_title("Feature distribution by block (schematic)\nTotal raw length 263 in repo extractor")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

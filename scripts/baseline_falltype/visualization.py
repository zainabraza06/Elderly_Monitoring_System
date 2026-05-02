"""Plots for multi-sensor fall-type classification."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

from .config import FALL_NAMES, SENSOR_FEATURE_PIE


def save_all_figures(
    results_dir: Path,
    label_encoder: LabelEncoder,
    y_test: np.ndarray,
    best_pred: np.ndarray,
    results: dict[str, dict],
    best_model_name: str,
    best_accuracy: float,
    y_encoded: np.ndarray,
) -> None:
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    short_names = list(label_encoder.classes_)
    labels_cm = [f"{code}\n({FALL_NAMES.get(code, code)})" for code in short_names]

    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="RdYlGn",
        ax=ax,
        xticklabels=labels_cm,
        yticklabels=labels_cm,
    )
    ax.set_title(
        f"Confusion Matrix - {best_model_name}\nAccuracy: {best_accuracy * 100:.1f}%",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    models_names = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in models_names]
    f1_scores = [results[m]["f1"] * 100 for m in models_names]
    x = np.arange(len(models_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#4CAF50")
    ax.bar(x + width / 2, f1_scores, width, label="Weighted F1", color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(models_names, rotation=45, ha="right")
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.set_ylim(0, 100)
    for i in range(len(models_names)):
        ax.text(i - width / 2, accuracies[i] + 1, f"{accuracies[i]:.1f}", ha="center", fontsize=9)
        ax.text(i + width / 2, f1_scores[i] + 1, f"{f1_scores[i]:.1f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(12, 6))
    xi = np.arange(len(short_names))
    w = 0.25
    ax.bar(xi - w, precision, w, label="Precision", color="#66BB6A")
    ax.bar(xi, recall, w, label="Recall", color="#42A5F5")
    ax.bar(xi + w, f1, w, label="F1-Score", color="#FF7043")
    ax.set_xticks(xi)
    ax.set_xticklabels([f"{c}\n{FALL_NAMES.get(c, c)}" for c in short_names], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance")
    ax.legend()
    ax.set_ylim(0, 1)
    for i, (p, r, fv) in enumerate(zip(precision, recall, f1)):
        ax.text(i - w, p + 0.02, f"{p:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w, fv + 0.02, f"{fv:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(figures_dir / "per_class_performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    n_classes = len(short_names)
    class_counts = [int(np.sum(y_encoded == i)) for i in range(n_classes)]
    colors_pie = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]
    ax.pie(
        class_counts,
        labels=[f"{c}\n{FALL_NAMES.get(c, c)}" for c in short_names],
        autopct="%1.1f%%",
        colors=colors_pie[:n_classes],
        explode=[0.02] * n_classes,
    )
    ax.set_title("Fall Type Distribution")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    total_feat = sum(SENSOR_FEATURE_PIE.values())
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
    ax.pie(
        SENSOR_FEATURE_PIE.values(),
        labels=SENSOR_FEATURE_PIE.keys(),
        autopct="%1.1f%%",
        colors=colors,
        explode=[0.02] * len(SENSOR_FEATURE_PIE),
    )
    ax.set_title(
        f"Feature distribution by sensor block\nTotal: {total_feat} features (before truncation to 350)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "sensor_contribution.png", dpi=300, bbox_inches="tight")
    plt.close()

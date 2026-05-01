import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_comparison(
    ml_fall_results,
    dl_fall_results,
    ml_adl_results,
    dl_adl_results,
    results_dir,
):
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle("MobiAct: Complete Model Comparison (Tasks 1 and 2)", fontsize=16, fontweight="bold")

    ax1 = plt.subplot(3, 4, 1)
    best_fall_ml = ml_fall_results[0]
    cm_fall = confusion_matrix(best_fall_ml["y_true"], best_fall_ml["y_pred"])
    sns.heatmap(
        cm_fall,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax1,
        xticklabels=["ADL", "Fall"],
        yticklabels=["ADL", "Fall"],
    )
    ax1.set_title(
        "Fall Detection - {}\nAcc={:.3f}".format(
            best_fall_ml["Model"], best_fall_ml["Accuracy"]
        )
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")

    ax2 = plt.subplot(3, 4, 2)
    best_fall_dl = dl_fall_results[3] if dl_fall_results[3]["F1"] > dl_fall_results[4]["F1"] else dl_fall_results[4]
    cm_fall_dl = confusion_matrix(best_fall_dl["y_true"], best_fall_dl["y_pred"])
    sns.heatmap(
        cm_fall_dl,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax2,
        xticklabels=["ADL", "Fall"],
        yticklabels=["ADL", "Fall"],
    )
    ax2.set_title(
        "Fall Detection - {}\nAcc={:.3f}".format(
            best_fall_dl["Model"], best_fall_dl["Accuracy"]
        )
    )
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    ax3 = plt.subplot(3, 4, 3)
    best_adl_ml = ml_adl_results[0]
    cm_adl = confusion_matrix(best_adl_ml["y_true"], best_adl_ml["y_pred"])
    sns.heatmap(cm_adl, annot=False, fmt="d", cmap="Reds", ax=ax3, cbar=False)
    ax3.set_title(
        "ADL Classification - {}\nAcc={:.3f}".format(
            best_adl_ml["Model"], best_adl_ml["Accuracy"]
        )
    )
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")

    ax4 = plt.subplot(3, 4, 4)
    best_adl_dl = dl_adl_results[4] if dl_adl_results[4]["F1"] > dl_adl_results[3]["F1"] else dl_adl_results[3]
    cm_adl_dl = confusion_matrix(best_adl_dl["y_true"], best_adl_dl["y_pred"])
    sns.heatmap(cm_adl_dl, annot=False, fmt="d", cmap="Purples", ax=ax4, cbar=False)
    ax4.set_title(
        "ADL Classification - {}\nAcc={:.3f}".format(
            best_adl_dl["Model"], best_adl_dl["Accuracy"]
        )
    )
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Actual")

    ax5 = plt.subplot(3, 4, 5)
    models_names = [r["Model"] for r in ml_fall_results] + [r["Model"] for r in dl_fall_results]
    fall_f1 = [r["F1"] for r in ml_fall_results] + [r["F1"] for r in dl_fall_results]
    adl_f1 = [r["F1"] for r in ml_adl_results] + [r["F1"] for r in dl_adl_results]
    x = np.arange(len(models_names))
    width = 0.35
    ax5.bar(x - width / 2, fall_f1, width, label="Fall Detection", color="#FF6B6B")
    ax5.bar(x + width / 2, adl_f1, width, label="ADL Classification", color="#4ECDC4")
    ax5.set_xticks(x)
    ax5.set_xticklabels(models_names, rotation=45, ha="right", fontsize=8)
    ax5.set_ylabel("F1 Score")
    ax5.set_title("F1 Score Comparison")
    ax5.legend()
    ax5.set_ylim(0, 1)

    ax6 = plt.subplot(3, 4, 6)
    ml_inference = [r["Inference_Time_ms"] for r in ml_fall_results]
    dl_inference = [r["Inference_Time_ms"] for r in dl_fall_results]
    all_inference = ml_inference + dl_inference
    ax6.bar(models_names, all_inference, color="#45B7D1")
    ax6.set_xticklabels(models_names, rotation=45, ha="right", fontsize=8)
    ax6.set_ylabel("Inference Time (ms/sample)")
    ax6.set_title("Inference Latency Comparison")
    ax6.set_yscale("log")
    ax6.axhline(y=1, color="r", linestyle="--", label="Mobile Target (<1ms)")
    ax6.legend()

    ax7 = plt.subplot(3, 4, 7)
    ml_sizes = [r["Model_Size_MB"] for r in ml_fall_results]
    dl_sizes = [r["Model_Size_MB"] for r in dl_fall_results]
    all_sizes = ml_sizes + dl_sizes
    colors = ["#2ecc71"] * len(ml_sizes) + ["#e74c3c"] * len(dl_sizes)
    ax7.bar(models_names, all_sizes, color=colors)
    ax7.set_xticklabels(models_names, rotation=45, ha="right", fontsize=8)
    ax7.set_ylabel("Model Size (MB)")
    ax7.set_title("Model Size Comparison")
    ax7.set_yscale("log")
    ax7.axhline(y=5, color="r", linestyle="--", label="Mobile Target (<5MB)")
    ax7.legend()

    ax8 = plt.subplot(3, 4, 8)
    ml_memory = [r["Memory_Usage_MB"] for r in ml_fall_results]
    dl_memory = [r["Memory_Usage_MB"] for r in dl_fall_results]
    all_memory = ml_memory + dl_memory
    ax8.bar(models_names, all_memory, color="#F39C12")
    ax8.set_xticklabels(models_names, rotation=45, ha="right", fontsize=8)
    ax8.set_ylabel("Memory Usage (MB)")
    ax8.set_title("Training Memory Usage")
    ax8.set_yscale("log")

    ax9 = plt.subplot(3, 4, 9)
    ml_time = [r["Train_Time_s"] for r in ml_fall_results]
    dl_time = [r["Train_Time_s"] for r in dl_fall_results]
    all_time = ml_time + dl_time
    ax9.bar(models_names, all_time, color="#9B59B6")
    ax9.set_xticklabels(models_names, rotation=45, ha="right", fontsize=8)
    ax9.set_ylabel("Training Time (seconds)")
    ax9.set_title("Training Time Comparison")
    ax9.set_yscale("log")

    ax10 = plt.subplot(3, 4, 10)
    dl_params = [r["Num_Params"] for r in dl_fall_results]
    dl_names = [r["Model"] for r in dl_fall_results]
    ax10.bar(dl_names, dl_params, color="#1ABC9C")
    ax10.set_xticklabels(dl_names, rotation=45, ha="right", fontsize=8)
    ax10.set_ylabel("Number of Parameters")
    ax10.set_title("DL Model Complexity")
    ax10.set_yscale("log")

    ax11 = plt.subplot(3, 4, 11)
    for r in ml_fall_results:
        ax11.scatter(
            r["Model_Size_MB"],
            r["Accuracy"],
            s=200,
            marker="o",
            label=f"ML-{r['Model']}",
            color="#3498db",
        )
    for r in dl_fall_results:
        ax11.scatter(
            r["Model_Size_MB"],
            r["Accuracy"],
            s=200,
            marker="s",
            label=f"DL-{r['Model']}",
            color="#e74c3c",
        )
    ax11.set_xlabel("Model Size (MB)")
    ax11.set_ylabel("Accuracy")
    ax11.set_title("Accuracy vs Model Size Trade-off")
    ax11.set_xscale("log")
    ax11.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax11.grid(True, alpha=0.3)

    ax12 = plt.subplot(3, 4, 12)
    ax12.axis("tight")
    ax12.axis("off")
    summary_data = [
        ["Metric", "Best ML", "Best DL"],
        [
            "Fall Detection F1",
            f"{best_fall_ml['F1']:.3f}",
            f"{best_fall_dl['F1']:.3f}",
        ],
        [
            "Fall Detection Acc",
            f"{best_fall_ml['Accuracy']:.3f}",
            f"{best_fall_dl['Accuracy']:.3f}",
        ],
        ["ADL F1", f"{best_adl_ml['F1']:.3f}", f"{best_adl_dl['F1']:.3f}"],
        [
            "ADL Acc",
            f"{best_adl_ml['Accuracy']:.3f}",
            f"{best_adl_dl['Accuracy']:.3f}",
        ],
        [
            "Inference (ms)",
            f"{best_fall_ml['Inference_Time_ms']:.3f}",
            f"{best_fall_dl['Inference_Time_ms']:.3f}",
        ],
        [
            "Model Size (MB)",
            f"{best_fall_ml['Model_Size_MB']:.1f}",
            f"{best_fall_dl['Model_Size_MB']:.1f}",
        ],
    ]
    table = ax12.table(cellText=summary_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax12.set_title("Best Model Comparison")

    plt.tight_layout()
    output_path = os.path.join(results_dir, "complete_model_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return output_path

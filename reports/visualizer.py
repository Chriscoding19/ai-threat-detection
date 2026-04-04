"""
visualizer.py
-------------
Generates ML evaluation plots and saves them to reports/figures/.

Plots generated:
  - Confusion matrix (per model)
  - Feature importance (Random Forest)
  - Class distribution
  - Model accuracy comparison bar chart
  - Anomaly score distribution (Isolation Forest)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

PALETTE = {
    "normal":             "#3fb950",
    "dos":                "#f85149",
    "port_scan":          "#d29922",
    "data_exfiltration":  "#388bfd",
    "brute_force":        "#bc8cff",
}

FIG_DIR = "reports/figures"


def _ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_class_distribution(label_series, title="Class Distribution"):
    """Bar chart of label counts."""
    _ensure_dir()
    counts = label_series.value_counts()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = [PALETTE.get(c, "#58a6ff") for c in counts.index]
    bars    = ax.bar(counts.index, counts.values, color=colors, edgecolor="#30363d", linewidth=0.8)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha="center", va="bottom", fontsize=10, color="#c9d1d9")

    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.set_title(title, color="#e6edf3", fontsize=14, fontweight="bold")
    ax.set_xlabel("Traffic Class", color="#8b949e")
    ax.set_ylabel("Count",         color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.spines[["top","right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    path = os.path.join(FIG_DIR, "class_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_confusion_matrix(cm: list, labels: list, model_name: str):
    """Heatmap confusion matrix."""
    _ensure_dir()
    cm_arr = np.array(cm)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="#30363d",
        ax=ax,
    )
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#161b22")
    ax.set_title(f"Confusion Matrix — {model_name}",
                 color="#e6edf3", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", color="#8b949e")
    ax.set_ylabel("True Label",      color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=9)

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
    path = os.path.join(FIG_DIR, f"confusion_{safe_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_feature_importance(importances: np.ndarray, feature_names: list,
                             top_n: int = 15):
    """Horizontal bar chart of top-N feature importances."""
    _ensure_dir()
    idx   = np.argsort(importances)[-top_n:]
    names = [feature_names[i] for i in idx]
    vals  = importances[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors  = plt.cm.Blues(np.linspace(0.4, 0.9, len(vals)))
    ax.barh(names, vals, color=colors, edgecolor="#30363d", linewidth=0.5)

    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.set_title("Feature Importance — Random Forest",
                 color="#e6edf3", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance", color="#8b949e")
    ax.tick_params(colors="#8b949e", labelsize=9)
    ax.spines[["top","right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    path = os.path.join(FIG_DIR, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_model_comparison(results: list):
    """Bar chart comparing accuracy and F1 across models."""
    _ensure_dir()
    names    = [r["model"] for r in results if "accuracy" in r]
    accs     = [r["accuracy"] for r in results if "accuracy" in r]
    f1s      = [r.get("f1_weighted", r.get("f1_binary", 0)) for r in results if "accuracy" in r]

    x    = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, accs, width, label="Accuracy", color="#388bfd", alpha=0.85)
    b2 = ax.bar(x + width/2, f1s,  width, label="F1 Score",  color="#3fb950", alpha=0.85)

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=8, color="#c9d1d9")

    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.set_title("Model Comparison — Accuracy & F1 Score",
                 color="#e6edf3", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" (RF + GB + IsoForest)", "\n(Ensemble)") for n in names],
                        color="#8b949e", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.spines[["top","right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    path = os.path.join(FIG_DIR, "model_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_anomaly_scores(scores: np.ndarray, y_true: np.ndarray,
                         label_names: list):
    """Distribution of Isolation Forest anomaly scores by class."""
    _ensure_dir()
    fig, ax = plt.subplots(figsize=(10, 5))

    for class_idx, name in enumerate(label_names):
        mask = y_true == class_idx
        if mask.sum() == 0:
            continue
        color = PALETTE.get(name, "#58a6ff")
        ax.hist(scores[mask], bins=40, alpha=0.6, color=color,
                label=name, edgecolor="none")

    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.set_title("Anomaly Score Distribution by Class (Isolation Forest)",
                 color="#e6edf3", fontsize=13, fontweight="bold")
    ax.set_xlabel("Anomaly Score (higher = more anomalous)", color="#8b949e")
    ax.set_ylabel("Count", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    ax.spines[["top","right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    path = os.path.join(FIG_DIR, "anomaly_scores.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
    return path

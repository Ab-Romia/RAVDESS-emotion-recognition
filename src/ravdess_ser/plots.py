"""Render the figures used in the README and the write-up from a results.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Dark-mode styling so figures sit natively on the site's dark page next to the
# hand-drawn SVG diagrams (transparent background, muted text, emerald accent).
_MUTED = "#a1a1aa"
_ACCENT = "#34d399"
plt.rcParams.update({
    "text.color": _MUTED, "axes.labelcolor": _MUTED, "axes.titlecolor": "#e4e4e7",
    "xtick.color": _MUTED, "ytick.color": _MUTED, "axes.edgecolor": "#3f3f46",
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "font.size": 11,
})
_EMERALD = LinearSegmentedColormap.from_list("emerald", ["#0c0c0f", "#0f766e", "#34d399"])


def plot_confusion_matrix(summary: dict, out_path: str | Path):
    cm = np.array(summary["confusion_matrix_total"], dtype=float)
    emotions = summary["emotions"]
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm_norm, cmap=_EMERALD, vmin=0, vmax=1)
    ax.set_xticks(range(len(emotions)))
    ax.set_yticks(range(len(emotions)))
    ax.set_xticklabels(emotions, rotation=45, ha="right")
    ax.set_yticklabels(emotions)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    acc = summary["accuracy_mean"]
    ax.set_title(f"RAVDESS speech, speaker-independent\n"
                 f"row-normalized confusion ({acc:.1%} mean accuracy)")
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            v = cm_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="#0a0a0b" if v > 0.55 else _MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)


def plot_per_emotion_f1(summary: dict, out_path: str | Path):
    per = summary["per_emotion_f1_mean"]
    emotions = list(per.keys())
    scores = [per[e] for e in emotions]
    order = np.argsort(scores)
    emotions = [emotions[i] for i in order]
    scores = [scores[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.barh(emotions, scores, color=_ACCENT)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (mean across folds)")
    ax.set_title("Per-emotion F1, speaker-independent cross-validation")
    for i, s in enumerate(scores):
        ax.text(s + 0.01, i, f"{s:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)


def plot_comparison(entries: list[tuple[str, float, float]], out_path: str | Path,
                    title: str, xlabel: str = "Accuracy"):
    """Horizontal bar chart of (label, mean, std) entries, drawn in input order."""
    labels = [e[0] for e in entries][::-1]
    means = [e[1] for e in entries][::-1]
    stds = [e[2] for e in entries][::-1]
    def _bar_color(l):
        low = l.lower()
        if "leak" in low or "random" in low:
            return "#ef4444"          # red: the inflated, leaky number
        if "fine-tun" in low or "overfit" in low:
            return "#71717A"          # gray: tried and rejected (overfits)
        return _ACCENT                # emerald: the honest progression
    colors = [_bar_color(l) for l in labels]
    fig, ax = plt.subplots(figsize=(7.4, 0.7 * len(labels) + 1.6))
    ax.barh(labels, means, xerr=stds, color=colors, capsize=4,
            error_kw={"ecolor": _MUTED})
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + (s or 0) + 0.01, i, f"{m:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)


def render_all(results_json: str | Path, out_dir: str | Path | None = None):
    results_json = Path(results_json)
    out_dir = Path(out_dir) if out_dir else results_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads(results_json.read_text())
    plot_confusion_matrix(summary, out_dir / "confusion_matrix.png")
    plot_per_emotion_f1(summary, out_dir / "per_emotion_f1.png")
    return out_dir


if __name__ == "__main__":
    import sys
    render_all(sys.argv[1])
    print("figures written")

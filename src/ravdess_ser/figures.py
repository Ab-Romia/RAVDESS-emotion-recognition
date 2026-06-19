"""Interpretation figures for the write-up, rendered from the run artifacts.

Everything here is derived from files the training runs already produced, so the
figures stay honest and reproducible:

* ``results/<model>/results.json`` : per-fold accuracy, per-emotion scores and,
  for late fusion, the per-fold temperatures and the selected audio weight.
* ``results/<model>_run.log`` / ``results/experiments.log`` : the per-epoch
  ``val_acc`` / ``val_macroF1`` traces and the held-out ``test acc`` lines.
* ``results/deploy/wavlm_head.pt`` : the trained head, including the learnable
  softmax weights over WavLM's 25 layers.

Run ``python -m ravdess_ser.figures results <out_dir>`` to regenerate them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Same dark-mode palette as plots.py so these sit natively on the site's dark
# page next to the hand-drawn SVG diagrams.
_MUTED = "#a1a1aa"
_FAINT = "#52525b"
_TEXT = "#e4e4e7"
_ACCENT = "#34d399"      # emerald: the honest result
_ACCENT_DEEP = "#0d9488" # teal-emerald: a second honest series
_TEAL = "#22d3ee"        # cyan: the visual modality
_GRAY = "#71717a"        # tried and rejected (fine-tuning overfits)
_RED = "#ef4444"         # the inflated, leaky number
_EDGE = "#3f3f46"
plt.rcParams.update({
    "text.color": _MUTED, "axes.labelcolor": _MUTED, "axes.titlecolor": _TEXT,
    "xtick.color": _MUTED, "ytick.color": _MUTED, "axes.edgecolor": _EDGE,
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "font.size": 11, "axes.grid": False,
})
_EMERALD = LinearSegmentedColormap.from_list("emerald", ["#0c0c0f", "#0f766e", "#34d399"])

_EPOCH_RE = re.compile(
    r"epoch\s+(\d+)/\d+\s+train_loss=([0-9.]+)\s+val_acc=([0-9.]+)\s+val_macroF1=([0-9.]+)")
_FOLD_RE = re.compile(r"\|\s*fold\s+(\d+)\s*\|")


def parse_run_log(path: str | Path) -> dict[int, list[dict]]:
    """Return {fold_index: [{epoch, train_loss, val_acc, val_f1}, ...]} from a run log."""
    folds: dict[int, list[dict]] = {}
    cur: int | None = None
    for line in Path(path).read_text(errors="ignore").splitlines():
        fold = _FOLD_RE.search(line)
        if fold:
            cur = int(fold.group(1))
            folds.setdefault(cur, [])
            continue
        ep = _EPOCH_RE.search(line)
        if ep and cur is not None:
            folds[cur].append({
                "epoch": int(ep.group(1)), "train_loss": float(ep.group(2)),
                "val_acc": float(ep.group(3)), "val_f1": float(ep.group(4)),
            })
    return {k: v for k, v in folds.items() if v}


def _mean_curve(folds: dict[int, list[dict]], key: str):
    """Per-epoch mean across folds still running at that epoch (n shrinks at the tail)."""
    max_ep = max(f[-1]["epoch"] for f in folds.values())
    xs, ys = [], []
    for e in range(1, max_ep + 1):
        vals = [pt[key] for f in folds.values() for pt in f if pt["epoch"] == e]
        if vals:
            xs.append(e)
            ys.append(float(np.mean(vals)))
    return np.array(xs), np.array(ys)


def _panel_curves(ax, folds, color, label, test_acc=None):
    for f in folds.values():
        x = [p["epoch"] for p in f]
        y = [p["val_f1"] for p in f]
        ax.plot(x, y, color=color, alpha=0.22, lw=1.1)
        ax.plot(x[-1], y[-1], "o", color=color, alpha=0.5, ms=4)  # early-stop point
    mx, my = _mean_curve(folds, "val_f1")
    ax.plot(mx, my, color=color, lw=2.6, label=label, solid_capstyle="round")
    ax.set_xlabel("epoch")
    ax.set_ylim(0.05, 0.85)
    ax.set_xlim(0.5, max(p["epoch"] for f in folds.values() for p in f) + 0.5)
    if test_acc is not None:
        ax.text(0.97, 0.06, f"held-out test: {test_acc:.1%}", transform=ax.transAxes,
                ha="right", va="bottom", color=_TEXT, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="#18181b", ec=_EDGE))


def plot_training_curves(frozen_log, finetune_log, frozen_test, finetune_test, out_path):
    """Two panels: validation macro-F1 per epoch for the frozen probe vs full fine-tune."""
    frozen = parse_run_log(frozen_log)
    finetune = parse_run_log(finetune_log)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    _panel_curves(a1, frozen, _ACCENT, "frozen mean", frozen_test)
    a1.set_title("Frozen WavLM + trained head")
    a1.set_ylabel("validation macro-F1")
    _panel_curves(a2, finetune, _GRAY, "fine-tune mean", finetune_test)
    a2.set_title("Full fine-tune (backbone unfrozen)")
    for ax in (a1, a2):
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Each thin line is one of 6 actor-disjoint folds; bold line is the mean",
                 color=_MUTED, fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)


def plot_layer_weights(ckpt_path, out_path):
    """Softmax of the learnable per-layer weights the head placed over WavLM's 25 layers."""
    import torch
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    raw = sd["encoder.layer_weights"].float().numpy()
    w = np.exp(raw - raw.max())
    w = w / w.sum()
    n = len(w)
    norm = (w - w.min()) / (w.max() - w.min() + 1e-9)
    colors = [_EMERALD(0.35 + 0.65 * v) for v in norm]

    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.bar(range(n), w, color=colors, width=0.82)
    ax.axhline(1 / n, color=_FAINT, ls="--", lw=1)
    ax.text(0.2, 1 / n + 0.0012, "uniform (no weighting)", color=_FAINT, va="bottom", fontsize=9)
    top = int(w.argmax())
    ax.annotate(f"mid-network layers (8-13) carry the most\nemotion (peak at layer {top}); the top layers,\nwhere speaker identity concentrates, fall below uniform",
                xy=(top, w[top]), xytext=(15.5, 0.049), ha="left", va="center",
                color=_TEXT, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=_MUTED, lw=1.2,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xlabel("WavLM layer  (0 = CNN features  →  24 = top transformer layer)")
    ax.set_ylabel("learned weight (softmax)")
    ax.set_title("What the model listens to: learned weighting over WavLM's 25 layers",
                 pad=14)
    ax.set_xticks(range(0, n, 2))
    ax.set_ylim(0, w.max() * 1.16)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)


def plot_fold_spread(entries, out_path):
    """Dot-per-fold accuracy spread for each model: shows variance, not just the mean.

    entries: list of (label, accs:list[float], color) in display order (left to right).
    """
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    rng = np.linspace(-0.12, 0.12, 8)
    for i, (label, accs, color) in enumerate(entries):
        jit = rng[: len(accs)]
        ax.scatter([i + j for j in jit], accs, color=color, s=46, alpha=0.85,
                   zorder=3, edgecolor="#0a0a0b", linewidth=0.5)
        m, s = float(np.mean(accs)), float(np.std(accs))
        ax.plot([i - 0.22, i + 0.22], [m, m], color=color, lw=2.6, zorder=4)
        ax.text(i, max(accs) + 0.022, f"{m:.1%}", ha="center", color=_TEXT, fontsize=10)
        ax.text(i, min(accs) - 0.030, f"±{s*100:.1f}", ha="center", color=_MUTED, fontsize=8)
    ax.axhline(0.125, color=_FAINT, ls="--", lw=1)
    ax.text(len(entries) - 0.5, 0.125, " chance (1/8)", color=_FAINT, va="bottom", ha="right", fontsize=9)
    ax.set_xticks(range(len(entries)))
    ax.set_xticklabels([e[0] for e in entries])
    ax.set_ylim(0.0, 0.95)
    ax.set_ylabel("accuracy per fold")
    ax.set_title("Honest spread: every fold plotted, 6 actor-disjoint splits per model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)


def plot_per_emotion_f1(summary, out_path, title):
    per = summary["per_emotion_f1_mean"]
    emotions = list(per.keys())
    scores = [per[e] for e in emotions]
    order = np.argsort(scores)
    emotions = [emotions[i] for i in order]
    scores = [scores[i] for i in order]
    colors = [_EMERALD(0.4 + 0.6 * s) for s in scores]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.barh(emotions, scores, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (mean across 6 folds)")
    ax.set_title(title)
    for i, s in enumerate(scores):
        ax.text(s + 0.012, i, f"{s:.2f}", va="center", fontsize=9, color=_TEXT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)


def plot_fusion_calibration(late_fusion, out_path):
    """Per-fold temperatures and selected audio weight: how late fusion is calibrated."""
    folds = late_fusion["folds"]
    idx = [f["fold"] for f in folds]
    ta = [f["temp_audio"] for f in folds]
    tv = [f["temp_visual"] for f in folds]
    wa = [f["fusion_weight_audio"] for f in folds]
    x = np.arange(len(idx))
    w = 0.38

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    ax.bar(x - w / 2, ta, w, color=_ACCENT, label="audio temperature")
    ax.bar(x + w / 2, tv, w, color=_TEAL, label="visual temperature")
    ax.axhline(1.0, color=_FAINT, ls="--", lw=1)
    ax.text(len(idx) - 0.5, 1.0, " T=1: no softening", color=_FAINT, va="bottom", ha="right", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"fold {i}" for i in idx])
    ax.set_ylabel("temperature (>1 softens overconfident logits)")
    ax.set_ylim(0, 2.3)
    ax.set_title("Each network is temperature-scaled on validation")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    ax2.bar(x, wa, 0.5, color=_ACCENT, label="audio")
    ax2.bar(x, [1 - v for v in wa], 0.5, bottom=wa, color=_TEAL, label="visual")
    for i, v in zip(x, wa):
        ax2.text(i, v / 2, f"{v:.2f}", ha="center", va="center", color="#0a0a0b", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i) for i in idx])
    ax2.set_xlabel("fold")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("fusion weight")
    ax2.set_title("Audio vs visual weight\n(grid-searched on validation)")
    ax2.legend(loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)


def plot_comparison(entries, out_path, title="Speaker-independent accuracy", xlabel="accuracy"):
    """Horizontal bars of (label, mean, std, color) in input order (top to bottom)."""
    labels = [e[0] for e in entries][::-1]
    means = [e[1] for e in entries][::-1]
    stds = [e[2] for e in entries][::-1]
    colors = [e[3] for e in entries][::-1]
    fig, ax = plt.subplots(figsize=(8.4, 0.62 * len(labels) + 1.5))
    ax.barh(labels, means, xerr=stds, color=colors, capsize=4, error_kw={"ecolor": _MUTED})
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + (s or 0) + 0.012, i, f"{m:.1%}", va="center", fontsize=9, color=_TEXT)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=True)
    plt.close(fig)


def render_blog_figures(results_dir, out_dir):
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load(name):
        return json.loads((results_dir / name / "results.json").read_text())

    hubert = load("hubert")
    frozen = load("wavlm_frozen")
    finetune = load("wavlm")
    fusion = load("late_fusion")

    # Leaky random-split HuBERT, parsed from the experiments log (same model, leaky split).
    leak_txt = (results_dir / "experiments.log").read_text(errors="ignore")
    leak_accs = [float(m) for m in re.findall(r"-> test acc=([0-9.]+)", leak_txt)]
    leak_mean, leak_std = float(np.mean(leak_accs)), float(np.std(leak_accs))

    plot_training_curves(
        results_dir / "wavlm_frozen_run.log", results_dir / "wavlm_run.log",
        frozen["accuracy_mean"], finetune["accuracy_mean"],
        out_dir / "training-curves.png")

    plot_layer_weights(results_dir / "deploy" / "wavlm_head.pt",
                       out_dir / "layer-weighting.png")

    plot_fold_spread([
        ("HuBERT-base", [f["accuracy"] for f in hubert["folds"]], _ACCENT),
        ("WavLM\nfine-tuned", [f["accuracy"] for f in finetune["folds"]], _GRAY),
        ("WavLM\nfrozen probe", [f["accuracy"] for f in frozen["folds"]], _ACCENT),
        ("Audio-visual\nfusion", [f["accuracy"] for f in fusion["folds"]], _ACCENT),
    ], out_dir / "per-fold-spread.png")

    plot_per_emotion_f1(fusion, out_dir / "per-emotion-f1.png",
                        "Per-emotion F1, audio-visual fusion (speaker-independent)")

    plot_fusion_calibration(fusion, out_dir / "fusion-calibration.png")

    plot_comparison([
        ("Audio-visual fusion", fusion["accuracy_mean"], fusion["accuracy_std"], _ACCENT),
        ("HuBERT, leaky random split", leak_mean, leak_std, _RED),
        ("WavLM frozen probe (audio)", frozen["accuracy_mean"], frozen["accuracy_std"], _ACCENT),
        ("WavLM fine-tuned (audio)", finetune["accuracy_mean"], finetune["accuracy_std"], _GRAY),
        ("HuBERT-base (audio)", hubert["accuracy_mean"], hubert["accuracy_std"], _ACCENT),
        ("Facial expression only", fusion["visual_only_mean"], 0.0, _ACCENT),
    ], out_dir / "model-comparison.png",
        title="Same model, honest vs leaky split, and the full stack")

    return out_dir


if __name__ == "__main__":
    import sys
    res = sys.argv[1] if len(sys.argv) > 1 else "results"
    out = sys.argv[2] if len(sys.argv) > 2 else "results/figures"
    render_blog_figures(res, out)
    print(f"figures written to {out}")

"""Speaker-independent k-fold cross-validation driver."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .audio import Collator, RavdessAudioDataset, load_resampled
from .config import Config, EMOTIONS
from .data import index_dataset, make_speaker_independent_folds, clips_for_actors
from .evaluate import run_inference, compute_metrics
from .models import SSLEmotionClassifier, SpectrogramCNN
from .train import class_weights, train_one_fold


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_fold_clip_lists(clips, cfg: Config, split_mode: str):
    """Return a list of (train, val, test) clip lists, one tuple per fold.

    `speaker_independent` builds folds from disjoint groups of actors (the honest
    protocol). `random` reproduces the original clip-level stratified split that
    ignores actor identity; it is included only to measure how much speaker leakage
    inflates the score, never as a result we stand behind.
    """
    if split_mode == "speaker_independent":
        folds = make_speaker_independent_folds(cfg)
        return [
            (clips_for_actors(clips, f.train_actors),
             clips_for_actors(clips, f.val_actors),
             clips_for_actors(clips, f.test_actors))
            for f in folds
        ]
    if split_mode == "random":
        from sklearn.model_selection import StratifiedKFold

        labels = np.array([c.emotion for c in clips])
        idx = np.arange(len(clips))
        skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True,
                              random_state=cfg.seed)
        rng = np.random.RandomState(cfg.seed)
        out = []
        for train_val_idx, test_idx in skf.split(idx, labels):
            tv = train_val_idx.copy()
            rng.shuffle(tv)
            n_val = max(1, int(0.15 * len(tv)))
            val_idx, tr_idx = tv[:n_val], tv[n_val:]
            out.append((
                [clips[i] for i in tr_idx],
                [clips[i] for i in val_idx],
                [clips[i] for i in test_idx],
            ))
        return out
    raise ValueError(f"unknown split_mode: {split_mode}")


def build_model(cfg: Config, model_type: str):
    if model_type in ("wavlm", "ssl"):
        return SSLEmotionClassifier(cfg)
    if model_type in ("wavlm_frozen", "ssl_frozen"):
        # Frozen linear-probe baseline: weighted-sum + attentive pooling + head over
        # frozen encoder features (the EmoBox-style setup), no encoder fine-tuning.
        frozen = replace(cfg, num_frozen_layers=999, grad_checkpointing=False)
        return SSLEmotionClassifier(frozen)
    if model_type == "spec_cnn":
        return SpectrogramCNN(cfg)
    raise ValueError(f"unknown model_type: {model_type}")


def _loader(clips, cfg, collate, shuffle):
    return DataLoader(
        RavdessAudioDataset(clips, cfg),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )


def run_crossval(cfg: Config, model_type: str = "hubert",
                 split_mode: str = "speaker_independent",
                 save_models: bool = True, log=print) -> dict:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clips = index_dataset(cfg.data_root)
    fold_splits = build_fold_clip_lists(clips, cfg, split_mode)

    # Warm the resample cache single-threaded so DataLoader workers never race
    # to write the same .npy file.
    log(f"warming resample cache for {len(clips)} clips...")
    for c in clips:
        load_resampled(c.path, cfg.sample_rate, cfg.cache_dir)

    train_collate = Collator(cfg, train=True)
    eval_collate = Collator(cfg, train=False)

    subdir = model_type if split_mode == "speaker_independent" else f"{model_type}_{split_mode}"
    out_dir = Path(cfg.results_dir) / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    cm_total = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=np.int64)

    for i, (train_clips, val_clips, test_clips) in enumerate(fold_splits):
        test_actors = sorted({c.actor for c in test_clips})
        log(f"\n=== {model_type} [{split_mode}] | fold {i} | "
            f"test actors {test_actors} ===")

        model = build_model(cfg, model_type)
        weights = class_weights(train_clips, device)
        model, best_val_f1 = train_one_fold(
            cfg, model,
            _loader(train_clips, cfg, train_collate, True),
            _loader(val_clips, cfg, eval_collate, False),
            device, weights=weights, log=log,
        )

        y_true, y_pred, _ = run_inference(
            model, _loader(test_clips, cfg, eval_collate, False), device)
        metrics = compute_metrics(y_true, y_pred)
        metrics["fold"] = i
        metrics["test_actors"] = test_actors
        metrics["best_val_macro_f1"] = best_val_f1
        fold_results.append(metrics)
        cm_total += np.array(metrics["confusion_matrix"])
        log(f"  -> test acc={metrics['accuracy']:.3f} "
            f"macroF1={metrics['macro_f1']:.3f}")

        if save_models:
            torch.save(model.state_dict(), out_dir / f"fold{i}.pt")
        del model
        torch.cuda.empty_cache()

    summary = _summarize(fold_results, cm_total, cfg, model_type)
    summary["split_mode"] = split_mode
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\n{model_type}: acc {summary['accuracy_mean']:.3f} "
        f"+/- {summary['accuracy_std']:.3f} | "
        f"macroF1 {summary['macro_f1_mean']:.3f} +/- {summary['macro_f1_std']:.3f}")
    return summary


def _summarize(fold_results, cm_total, cfg: Config, model_type: str) -> dict:
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]
    # Representative fold for deployment: closest test accuracy to the mean.
    mean_acc = float(np.mean(accs))
    rep = int(np.argmin([abs(a - mean_acc) for a in accs]))
    per_emotion_f1 = {
        e: float(np.mean([r["per_emotion"][e]["f1"] for r in fold_results]))
        for e in EMOTIONS
    }
    return {
        "model_type": model_type,
        "config": cfg.to_dict(),
        "num_folds": len(fold_results),
        "accuracy_mean": mean_acc,
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "per_emotion_f1_mean": per_emotion_f1,
        "representative_fold": rep,
        "confusion_matrix_total": cm_total.tolist(),
        "emotions": EMOTIONS,
        "folds": fold_results,
    }

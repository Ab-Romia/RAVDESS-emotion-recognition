"""Inference and metric computation."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import EMOTIONS


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for input_values, attention_mask, labels in loader:
        input_values = input_values.to(device)
        attention_mask = attention_mask.to(device)
        logits = model(input_values, attention_mask)
        probs = torch.softmax(logits, dim=1)
        y_prob.append(probs.cpu().numpy())
        y_pred.append(probs.argmax(1).cpu().numpy())
        y_true.append(labels.numpy())
    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_prob),
    )


def compute_metrics(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(EMOTIONS))), zero_division=0
    )
    per_emotion = {
        EMOTIONS[i]: {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(EMOTIONS))
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EMOTIONS))))
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_emotion": per_emotion,
        "confusion_matrix": cm.tolist(),
    }

"""Single-fold training loop with class weighting, AMP and early stopping."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight

from .config import Config, NUM_EMOTIONS
from .evaluate import run_inference, compute_metrics


def snapshot_to_cpu(model) -> dict:
    """Clone the model weights onto CPU RAM so the best checkpoint never holds GPU
    memory. Deep-copying a large state dict on the GPU is enough to cause OOM."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def class_weights(train_clips, device) -> torch.Tensor:
    labels = np.array([c.emotion for c in train_clips])
    classes = np.arange(NUM_EMOTIONS)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_fold(cfg: Config, model, train_loader, val_loader, device,
                   weights=None, log=print):
    from transformers import get_cosine_schedule_with_warmup

    model.to(device)
    optimizer = torch.optim.AdamW(model.param_groups(), weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=weights)
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Cosine schedule over the full training horizon so the learning rate actually
    # anneals; with early stopping the best checkpoint then lands in a low-LR regime.
    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1 = -1.0
    best_state = snapshot_to_cpu(model)
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, (input_values, attention_mask, labels) in enumerate(train_loader):
            input_values = input_values.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_values, attention_mask)
                loss = criterion(logits, labels) / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            running += loss.item() * cfg.grad_accum_steps

            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        y_true, y_pred, _ = run_inference(model, val_loader, device)
        val = compute_metrics(y_true, y_pred)
        log(f"  epoch {epoch + 1:2d}/{cfg.epochs}  "
            f"train_loss={running / len(train_loader):.3f}  "
            f"val_acc={val['accuracy']:.3f}  val_macroF1={val['macro_f1']:.3f}")

        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = snapshot_to_cpu(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                log(f"  early stop at epoch {epoch + 1} (best val macroF1={best_f1:.3f})")
                break

    model.load_state_dict(best_state)
    return model, best_f1

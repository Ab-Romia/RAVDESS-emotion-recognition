"""Genuine audio-visual emotion recognition.

Each speech clip is paired with the face track from its matching audio-visual
recording. The audio branch is identical to the audio-only model, so comparing
the two is a clean ablation of what the face actually adds. Per-frame visual
features come from a frozen ImageNet backbone and are cached once; only the audio
encoder and the fusion head are trained.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .audio import load_resampled, Collator
from .config import Config, EMOTIONS, NUM_EMOTIONS
from .data import index_dataset, make_speaker_independent_folds
from .evaluate import compute_metrics
from .models import SSLAudioEncoder
from .train import class_weights, snapshot_to_cpu
from .video import extract_face_frames


@dataclass(frozen=True)
class AVClip:
    audio_path: str
    video_path: str
    emotion: int
    actor: int


def _key(clip):
    # (emotion, intensity, statement, repetition, actor) identifies a performance.
    return (clip.emotion, clip.intensity, clip.statement, clip.repetition, clip.actor)


def index_av_clips(cfg: Config) -> list[AVClip]:
    """Pair every speech wav with the face video of the same performance."""
    audio = {_key(c): c for c in index_dataset(cfg.data_root)}
    pairs: list[AVClip] = []
    for mp4 in sorted(Path(cfg.video_root).rglob("01-01-*.mp4")):
        parts = mp4.stem.split("-")
        if len(parts) != 7:
            continue
        _, _, emo, inten, stmt, rep, actor = (int(p) for p in parts)
        key = (emo - 1, inten, stmt, rep, actor)
        match = audio.get(key)
        if match is not None:
            pairs.append(AVClip(audio_path=match.path, video_path=str(mp4),
                                emotion=emo - 1, actor=actor))
    if not pairs:
        raise RuntimeError(f"No audio-visual pairs found under {cfg.video_root}")
    return pairs


def _vfeat_path(cfg: Config, video_path: str) -> Path:
    tag = cfg.visual_backbone.split("/")[-1]
    return Path(cfg.cache_dir) / (Path(video_path).stem + f"_vfeat_{tag}.npy")


def _pool_visual(out):
    """One feature vector per frame, robust to CNN or transformer backbones.

    Avoids `pooler_output`: a ViT classification checkpoint leaves the pooler
    randomly initialized, so its pooler output is meaningless. The trained
    representation is the CLS token; for CNNs it is the spatial average.
    """
    h = out.last_hidden_state
    if h.dim() == 4:           # CNN: (B, C, H, W)
        return h.mean(dim=(2, 3))
    return h[:, 0]             # transformer: the CLS token


def precompute_visual_features(clips: list[AVClip], cfg: Config, device, log=print):
    """Run the frozen visual backbone over each clip's faces once and cache it."""
    from transformers import AutoImageProcessor, AutoModel

    todo = [c for c in clips if not _vfeat_path(cfg, c.video_path).exists()]
    if not todo:
        log("visual features already cached")
        return
    log(f"extracting visual features for {len(todo)} clips...")
    processor = AutoImageProcessor.from_pretrained(cfg.visual_backbone)
    backbone = AutoModel.from_pretrained(cfg.visual_backbone).to(device).eval()
    Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
    for i, clip in enumerate(todo):
        # Cache the face crops so swapping the visual backbone later only re-runs
        # the (fast) backbone forward, not the (slow) face extraction.
        frames = extract_face_frames(clip.video_path, cfg.num_frames,
                                     cfg.face_size, cache_dir=cfg.cache_dir)
        inputs = processor(images=list(frames), return_tensors="pt").to(device)
        with torch.no_grad():
            out = backbone(**inputs)
            feats = _pool_visual(out)
        np.save(_vfeat_path(cfg, clip.video_path),
                feats.cpu().numpy().astype(np.float32))
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(todo)}")
    del backbone
    torch.cuda.empty_cache()


class MultiModalDataset(Dataset):
    def __init__(self, clips: list[AVClip], cfg: Config):
        self.clips = clips
        self.cfg = cfg

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        wav = load_resampled(clip.audio_path, self.cfg.sample_rate, self.cfg.cache_dir)
        vfeat = np.load(_vfeat_path(self.cfg, clip.video_path))
        return wav, vfeat, clip.emotion


class MultiModalCollator:
    """Reuses the audio Collator (crop + train-only augmentation + features) and
    stacks the cached visual features alongside."""

    def __init__(self, cfg: Config, train: bool = False):
        self.audio = Collator(cfg, train=train)

    def __call__(self, batch):
        wavs, vfeats, labels = zip(*batch)
        rng = np.random.default_rng()
        fitted = [self.audio._prepare(w, rng) for w in wavs]
        inputs = self.audio.feature_extractor(
            fitted, sampling_rate=self.audio.cfg.sample_rate, return_tensors="pt",
            padding="longest", return_attention_mask=True)
        visual = torch.tensor(np.stack(vfeats), dtype=torch.float32)
        return (inputs.input_values, inputs.attention_mask, visual,
                torch.tensor(labels, dtype=torch.long))


class MultiModalNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.audio_encoder = SSLAudioEncoder(cfg)  # identical to the audio-only model
        d = cfg.fusion_dim
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_encoder.out_dim, d), nn.ReLU(), nn.Dropout(cfg.dropout))
        self.frame_score = nn.Sequential(
            nn.Linear(cfg.visual_feat_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        self.visual_proj = nn.Sequential(
            nn.Linear(cfg.visual_feat_dim, d), nn.ReLU(), nn.Dropout(cfg.dropout))
        self.gate = nn.Sequential(nn.Linear(2 * d, 2), nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(
            nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(d, NUM_EMOTIONS))

    def forward(self, input_values, attention_mask, visual):
        audio = self.audio_proj(self.audio_encoder(input_values, attention_mask))

        attn = torch.softmax(self.frame_score(visual), dim=1)  # (B, T, 1)
        visual = self.visual_proj((visual * attn).sum(1))

        g = self.gate(torch.cat([audio, visual], dim=1))
        fused = torch.cat([g[:, 0:1] * audio, g[:, 1:2] * visual], dim=1)
        return self.classifier(fused)

    def param_groups(self):
        groups = self.audio_encoder.lr_scaled_groups(
            self.cfg.lr_backbone, self.cfg.lr_head, self.cfg.llrd)
        fusion = (list(self.audio_proj.parameters()) + list(self.frame_score.parameters())
                  + list(self.visual_proj.parameters()) + list(self.gate.parameters())
                  + list(self.classifier.parameters()))
        groups.append({"params": fusion, "lr": self.cfg.lr_head})
        return groups


def _train_fold(cfg, model, train_loader, val_loader, device, weights, log):
    from transformers import get_cosine_schedule_with_warmup

    model.to(device)
    opt = torch.optim.AdamW(model.param_groups(), weight_decay=cfg.weight_decay)
    crit = nn.CrossEntropyLoss(weight=weights)
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        opt, int(cfg.warmup_ratio * total_steps), total_steps)
    best_f1, best_state, no_improve = -1.0, snapshot_to_cpu(model), 0

    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, (iv, am, vis, y) in enumerate(train_loader):
            iv, am, vis, y = iv.to(device), am.to(device), vis.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(model(iv, am, vis), y) / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
        yt, yp, _ = _infer(model, val_loader, device)
        val = compute_metrics(yt, yp)
        log(f"  epoch {epoch+1:2d}/{cfg.epochs}  val_acc={val['accuracy']:.3f} "
            f"val_macroF1={val['macro_f1']:.3f}")
        if val["macro_f1"] > best_f1:
            best_f1, best_state, no_improve = val["macro_f1"], snapshot_to_cpu(model), 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                log(f"  early stop (best val macroF1={best_f1:.3f})")
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _infer(model, loader, device):
    model.eval()
    yt, yp = [], []
    for iv, am, vis, y in loader:
        logits = model(iv.to(device), am.to(device), vis.to(device))
        yp.append(logits.argmax(1).cpu().numpy())
        yt.append(y.numpy())
    return np.concatenate(yt), np.concatenate(yp), None


def run_multimodal_crossval(cfg: Config, save_models=True, log=print) -> dict:
    from .crossval import seed_everything, _summarize
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clips = index_av_clips(cfg)
    log(f"paired {len(clips)} audio-visual clips")

    # Warm both caches before spinning up workers.
    for c in clips:
        load_resampled(c.audio_path, cfg.sample_rate, cfg.cache_dir)
    precompute_visual_features(clips, cfg, device, log=log)

    train_collate = MultiModalCollator(cfg, train=True)
    eval_collate = MultiModalCollator(cfg, train=False)
    out_dir = Path(cfg.results_dir) / "multimodal"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_actor = {}
    for c in clips:
        by_actor.setdefault(c.actor, []).append(c)

    fold_results, cm_total = [], np.zeros((NUM_EMOTIONS, NUM_EMOTIONS), dtype=np.int64)
    for fold in make_speaker_independent_folds(cfg):
        sel = lambda actors: [c for a in actors for c in by_actor.get(a, [])]
        train, val, test = sel(fold.train_actors), sel(fold.val_actors), sel(fold.test_actors)
        log(f"\n=== multimodal | fold {fold.index} | test actors {fold.test_actors} ===")
        mk = lambda cl, sh, cz: DataLoader(MultiModalDataset(cl, cfg), batch_size=cfg.batch_size,
                                           shuffle=sh, num_workers=cfg.num_workers,
                                           collate_fn=cz, pin_memory=True,
                                           persistent_workers=cfg.num_workers > 0)
        model = MultiModalNet(cfg)
        weights = class_weights(train, device)
        model = _train_fold(cfg, model, mk(train, True, train_collate),
                            mk(val, False, eval_collate), device, weights, log)
        yt, yp, _ = _infer(model, mk(test, False, eval_collate), device)
        metrics = compute_metrics(yt, yp)
        metrics["fold"] = fold.index
        metrics["test_actors"] = list(fold.test_actors)
        fold_results.append(metrics)
        cm_total += np.array(metrics["confusion_matrix"])
        log(f"  -> test acc={metrics['accuracy']:.3f} macroF1={metrics['macro_f1']:.3f}")
        if save_models:
            torch.save(model.state_dict(), out_dir / f"fold{fold.index}.pt")
        del model
        torch.cuda.empty_cache()

    summary = _summarize(fold_results, cm_total, cfg, "multimodal")
    summary["split_mode"] = "speaker_independent"
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nmultimodal: acc {summary['accuracy_mean']:.3f} +/- {summary['accuracy_std']:.3f} "
        f"| macroF1 {summary['macro_f1_mean']:.3f} +/- {summary['macro_f1_std']:.3f}")
    return summary

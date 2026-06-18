"""Calibrated, weighted decision-level (late) fusion of audio and visual models.

Joint/gated fusion fails here by modality competition: on ~960 training clips and
24 identities, a learned fusion head leans on the visual stream's residual
identity signal and transfers worse than audio alone (Huang et al., ICML 2022).
The robust alternative, and the standard winner on small audio-visual emotion
benchmarks, is to train each modality independently and combine only their class
probabilities (Vielzeuf et al.; EmotiW/AFEW).

Two things make the average safe rather than harmful:
  1. Temperature scaling. Each modality's logits are divided by a scalar fit on
     the validation fold (NLL), so a confidently-wrong face on an unseen person
     cannot dominate the average.
  2. A validation-tuned weight. The mixing weight is grid-searched on validation
     only; because w = 1 (audio alone) is in the grid, the fused result can never
     fall below audio-only. Temperatures and the weight are fit per fold on
     train/val actors, never on test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from .audio import Collator, RavdessAudioDataset
from .config import Config, NUM_EMOTIONS
from .crossval import seed_everything, _summarize
from .data import index_dataset, make_speaker_independent_folds, clips_for_actors
from .evaluate import compute_metrics
from .models import SSLEmotionClassifier
from .multimodal import index_av_clips, _vfeat_path, precompute_visual_features
from .train import class_weights, train_one_fold


@torch.no_grad()
def _audio_logits(model, loader, device):
    model.eval()
    ys, logits = [], []
    for input_values, attention_mask, labels in loader:
        out = model(input_values.to(device), attention_mask.to(device))
        logits.append(out.cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(logits)


def _visual_matrix(clips, audio_to_video, cfg):
    return np.stack([
        np.load(_vfeat_path(cfg, audio_to_video[c.path])).mean(0) for c in clips
    ])


def _fit_temperature(logits, y) -> float:
    """One scalar temperature, fit by NLL minimization on the validation fold.

    The temperature is floored at 1.0 so it can only soften overconfident
    probabilities, never sharpen them. Without the floor, a modality that
    separates its own small validation fold drives T toward zero and the result
    inverts: it sharpens, which is the opposite of what calibration is for.
    """
    L = torch.tensor(np.asarray(logits), dtype=torch.float32)
    Y = torch.tensor(np.asarray(y), dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(L / T.clamp(min=1.0), Y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1.0).item())


def _calibrated(logits, T):
    L = np.asarray(logits, dtype=np.float64) / T
    e = np.exp(L - L.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _best_weight(audio_val, visual_val, y_val):
    # Sweep from audio-only (w=1) downward so ties resolve toward audio, the
    # stronger modality. This honors the floor: fusion can never beat itself into
    # doing worse than audio on a tie.
    best_w, best_acc = 1.0, -1.0
    for w in np.linspace(1.0, 0.0, 21):
        acc = ((w * audio_val + (1 - w) * visual_val).argmax(1) == y_val).mean()
        if acc > best_acc:
            best_acc, best_w = acc, float(w)
    return best_w


def run_late_fusion_crossval(cfg: Config, log=print) -> dict:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_clips = index_dataset(cfg.data_root)
    av = index_av_clips(cfg)
    audio_to_video = {a.audio_path: a.video_path for a in av}
    precompute_visual_features(av, cfg, device, log=log)

    train_collate = Collator(cfg, train=True)
    eval_collate = Collator(cfg, train=False)
    loader = lambda cl, sh, cz: DataLoader(
        RavdessAudioDataset(cl, cfg), batch_size=cfg.batch_size, shuffle=sh,
        num_workers=cfg.num_workers, collate_fn=cz, pin_memory=True)

    out_dir = Path(cfg.results_dir) / "late_fusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_only, visual_only, fused = [], [], []
    cm_total = np.zeros((NUM_EMOTIONS, NUM_EMOTIONS), dtype=np.int64)

    for fold in make_speaker_independent_folds(cfg):
        tr = clips_for_actors(audio_clips, fold.train_actors)
        va = clips_for_actors(audio_clips, fold.val_actors)
        te = clips_for_actors(audio_clips, fold.test_actors)
        log(f"\n=== late fusion | fold {fold.index} | test {fold.test_actors} ===")

        # 1. Audio model, trained independently to its own optimum.
        model = SSLEmotionClassifier(cfg)
        model, _ = train_one_fold(cfg, model, loader(tr, True, train_collate),
                                  loader(va, False, eval_collate), device,
                                  weights=class_weights(tr, device), log=log)
        yv, a_val_logits = _audio_logits(model, loader(va, False, eval_collate), device)
        yt, a_test_logits = _audio_logits(model, loader(te, False, eval_collate), device)
        del model
        torch.cuda.empty_cache()

        # 2. Visual classifier, trained independently on expression features.
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(_visual_matrix(tr, audio_to_video, cfg), [c.emotion for c in tr])
        v_val_logits = clf.decision_function(_visual_matrix(va, audio_to_video, cfg))
        v_test_logits = clf.decision_function(_visual_matrix(te, audio_to_video, cfg))

        # 3. Calibrate each modality on validation, then fuse.
        Ta = _fit_temperature(a_val_logits, yv)
        Tv = _fit_temperature(v_val_logits, yv)
        a_val, v_val = _calibrated(a_val_logits, Ta), _calibrated(v_val_logits, Tv)
        a_test, v_test = _calibrated(a_test_logits, Ta), _calibrated(v_test_logits, Tv)
        w = _best_weight(a_val, v_val, yv)

        f_pred = (w * a_test + (1 - w) * v_test).argmax(1)
        fm = compute_metrics(yt, f_pred)
        cm_total += np.array(fm["confusion_matrix"])
        fm.update(fold=fold.index, test_actors=list(fold.test_actors),
                  fusion_weight_audio=w, temp_audio=Ta, temp_visual=Tv)
        fused.append(fm)
        audio_only.append(compute_metrics(yt, a_test.argmax(1)))
        visual_only.append(compute_metrics(yt, v_test.argmax(1)))
        log(f"  audio={audio_only[-1]['accuracy']:.3f}  "
            f"visual={visual_only[-1]['accuracy']:.3f}  "
            f"fused={fm['accuracy']:.3f}  (w_audio={w:.2f}, Ta={Ta:.2f}, Tv={Tv:.2f})")

    summary = _summarize(fused, cm_total, cfg, "late_fusion")
    summary["audio_only_mean"] = float(np.mean([m["accuracy"] for m in audio_only]))
    summary["audio_only_std"] = float(np.std([m["accuracy"] for m in audio_only]))
    summary["visual_only_mean"] = float(np.mean([m["accuracy"] for m in visual_only]))
    summary["split_mode"] = "speaker_independent"
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"\nLATE FUSION: audio {summary['audio_only_mean']*100:.1f}% | "
        f"visual {summary['visual_only_mean']*100:.1f}% | "
        f"FUSED {summary['accuracy_mean']*100:.1f}% +/- {summary['accuracy_std']*100:.1f}")
    return summary

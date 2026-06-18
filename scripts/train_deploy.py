"""Train a single deployment checkpoint on a fixed speaker-independent split.

The cross-validation reports the honest mean across folds; for the live demo we
need one saved model. This trains the frozen WavLM-large probe (the best audio
recipe) on 16 actors, validates on 4, and reports accuracy on 4 held-out actors
it never saw, then saves the weights for the Space.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from ravdess_ser.audio import Collator, RavdessAudioDataset
from ravdess_ser.config import Config
from ravdess_ser.crossval import seed_everything
from ravdess_ser.data import index_dataset, clips_for_actors
from ravdess_ser.evaluate import run_inference, compute_metrics
from ravdess_ser.models import SSLEmotionClassifier
from ravdess_ser.train import class_weights, train_one_fold

# Gender-balanced fixed split (odd = male, even = female).
TRAIN = list(range(1, 17))     # actors 1-16
VAL = [17, 18, 19, 20]         # held out for early stopping
TEST = [21, 22, 23, 24]        # never seen


def main():
    cfg = Config.from_yaml("configs/audio_wavlm.yaml")
    cfg = replace(cfg, num_frozen_layers=999, grad_checkpointing=False, epochs=30)
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clips = index_dataset(cfg.data_root)
    tr, va, te = (clips_for_actors(clips, a) for a in (TRAIN, VAL, TEST))

    tc, ec = Collator(cfg, train=True), Collator(cfg, train=False)
    mk = lambda cl, sh, cz: DataLoader(RavdessAudioDataset(cl, cfg),
                                       batch_size=cfg.batch_size, shuffle=sh,
                                       num_workers=cfg.num_workers, collate_fn=cz,
                                       pin_memory=True)
    model = SSLEmotionClassifier(cfg)
    weights = class_weights(tr, device)
    model, _ = train_one_fold(cfg, model, mk(tr, True, tc), mk(va, False, ec),
                              device, weights=weights)
    yt, yp, _ = run_inference(model, mk(te, False, ec), device)
    m = compute_metrics(yt, yp)
    print(f"held-out test acc={m['accuracy']:.3f} macroF1={m['macro_f1']:.3f}")

    out = Path("results/deploy")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "wavlm_frozen_deploy.pt")
    import json
    (out / "deploy_metrics.json").write_text(json.dumps(
        {"split": {"train": TRAIN, "val": VAL, "test": TEST}, **m}, indent=2))
    print(f"saved checkpoint to {out/'wavlm_frozen_deploy.pt'}")


if __name__ == "__main__":
    main()

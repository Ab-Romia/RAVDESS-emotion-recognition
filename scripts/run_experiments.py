"""Run the remaining experiments back-to-back after the primary model.

The primary result (wavlm, speaker-independent) is produced by run_crossval.py.
This driver adds, in order:
  1. wavlm under a random clip-level split  -> quantifies the speaker-leakage inflation
  2. wavlm_frozen, speaker-independent       -> frozen linear-probe baseline (no fine-tuning)
  3. spec_cnn, speaker-independent            -> from-scratch baseline (no pretraining)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ravdess_ser.config import Config
from ravdess_ser.crossval import run_crossval

CONFIG = "configs/audio_wavlm.yaml"
EXPERIMENTS = [
    ("wavlm", "random"),
    ("wavlm_frozen", "speaker_independent"),
    ("spec_cnn", "speaker_independent"),
]


def main():
    for model_type, split in EXPERIMENTS:
        print(f"\n########## {model_type} [{split}] ##########", flush=True)
        cfg = Config.from_yaml(CONFIG)
        # The frozen probe and the from-scratch CNN have no pretrained weights to
        # protect, so give them more epochs and a higher head learning rate.
        if model_type in ("spec_cnn", "wavlm_frozen"):
            cfg.epochs = 40
            cfg.lr_head = 2e-3
        run_crossval(cfg, model_type=model_type, split_mode=split,
                     save_models=(model_type == "wavlm"))


if __name__ == "__main__":
    main()

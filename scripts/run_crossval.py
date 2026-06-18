"""Run speaker-independent cross-validation for one model type.

Examples:
    python scripts/run_crossval.py --model wavlm --config configs/audio_wavlm.yaml
    python scripts/run_crossval.py --model wavlm_frozen --config configs/audio_wavlm.yaml
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ravdess_ser.config import Config
from ravdess_ser.crossval import run_crossval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="wavlm",
                        choices=["wavlm", "wavlm_frozen", "spec_cnn"])
    parser.add_argument("--split", default="speaker_independent",
                        choices=["speaker_independent", "random"],
                        help="random reproduces the leaky split, for comparison only")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.data_root:
        cfg.data_root = args.data_root
    if args.epochs:
        cfg.epochs = args.epochs

    run_crossval(cfg, model_type=args.model, split_mode=args.split,
                 save_models=not args.no_save_models)


if __name__ == "__main__":
    main()

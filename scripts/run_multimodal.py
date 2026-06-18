"""Run speaker-independent cross-validation for the audio-visual model.

    python scripts/run_multimodal.py --config configs/audio_wavlm.yaml
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ravdess_ser.config import Config
from ravdess_ser.multimodal import run_multimodal_crossval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/audio_wavlm.yaml")
    parser.add_argument("--video-root", default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.video_root:
        cfg.video_root = args.video_root
    run_multimodal_crossval(cfg)


if __name__ == "__main__":
    main()

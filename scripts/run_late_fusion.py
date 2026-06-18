"""Run calibrated, weighted late fusion of the audio and visual models.

    python scripts/run_late_fusion.py --config configs/multimodal_wavlm.yaml
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ravdess_ser.config import Config
from ravdess_ser.late_fusion import run_late_fusion_crossval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multimodal_wavlm.yaml")
    args = parser.parse_args()
    run_late_fusion_crossval(Config.from_yaml(args.config))


if __name__ == "__main__":
    main()

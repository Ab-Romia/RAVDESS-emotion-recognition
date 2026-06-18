"""Configuration and label definitions for RAVDESS speech emotion recognition."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

# RAVDESS emotion codes appear as the third field of each filename and are
# 1-indexed. We map them to contiguous 0-based class indices for training.
EMOTION_CODE_TO_NAME = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

# Class index (0-7) -> name, in the order the model emits logits.
EMOTIONS = [EMOTION_CODE_TO_NAME[c] for c in range(1, 9)]
NUM_EMOTIONS = len(EMOTIONS)


def actor_gender(actor_id: int) -> str:
    """RAVDESS actors are numbered so that odd ids are male and even are female."""
    return "male" if actor_id % 2 == 1 else "female"


@dataclass
class Config:
    # Data
    data_root: str = "/home/romia/ravdess-data/Audio_Speech"
    sample_rate: int = 16000
    max_seconds: float = 5.0  # covers the longest RAVDESS clips; shorter ones are padded

    # Model. The encoder is any wav2vec2/HuBERT/WavLM checkpoint; emotion signal
    # concentrates in the mid/upper layers, so a learnable weighted sum over all
    # transformer layers is used instead of the final layer alone.
    ssl_name: str = "microsoft/wavlm-large"
    freeze_feature_extractor: bool = True
    num_frozen_layers: int = 18  # of WavLM-large's 24 transformer layers (fine-tune top 6)
    use_layer_weighting: bool = True
    hidden_dim: int = 256
    dropout: float = 0.3

    # Cross-validation
    num_folds: int = 6  # 24 actors / 6 = 4 test actors per fold
    val_actors_per_fold: int = 4  # held out from the train actors for early stopping
    seed: int = 42

    # Training
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 18
    patience: int = 7
    warmup_ratio: float = 0.1
    lr_head: float = 1e-3
    lr_backbone: float = 5e-6
    llrd: float = 0.9  # layer-wise learning-rate decay across backbone layers
    weight_decay: float = 1e-4
    augment: bool = True
    use_amp: bool = True
    grad_checkpointing: bool = True
    num_workers: int = 4

    # Multimodal (video). The visual branch pairs each speech clip with the face
    # track from the matching audio-visual recording.
    video_root: str = "/home/romia/ravdess-data/video_extracted"
    num_frames: int = 16
    face_size: int = 224  # input resolution for the visual backbone
    visual_backbone: str = "google/mobilenet_v2_1.0_224"
    visual_feat_dim: int = 1280
    fusion_dim: int = 256

    # Output
    results_dir: str = "results"
    cache_dir: str = ".cache"

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = {k: v for k, v in data.items() if k not in cls.__dataclass_fields__}
        cfg = cls(**known)
        cfg.extra = unknown
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

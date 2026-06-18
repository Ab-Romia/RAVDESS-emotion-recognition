"""Single-clip inference, used by the demo app and the command line.

Loads a checkpoint into the exact training architecture and runs deterministic,
augmentation-free prediction. The checkpoint is loaded with strict=True so a
mismatch fails loudly instead of silently producing random output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .audio import load_resampled
from .config import Config, EMOTIONS
from .models import SSLEmotionClassifier


class EmotionPredictor:
    def __init__(self, checkpoint_path: str, cfg: Config | None = None, device=None):
        self.cfg = cfg or Config()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = SSLEmotionClassifier(self.cfg)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self._load(state)
        self.model.to(self.device).eval()

        from transformers import AutoFeatureExtractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.cfg.ssl_name)
        self.max_len = int(self.cfg.sample_rate * self.cfg.max_seconds)

    def _load(self, state):
        """Load a full checkpoint strictly, or a head-only checkpoint (the frozen
        backbone is already loaded from the hub). Either way, fail loudly: nothing
        unexpected, and the only thing a head-only file may omit is the backbone."""
        if any(k.startswith("encoder.backbone.") for k in state):
            self.model.load_state_dict(state, strict=True)
            return
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        non_backbone = [k for k in missing if not k.startswith("encoder.backbone.")]
        if unexpected or non_backbone:
            raise RuntimeError(
                f"checkpoint mismatch: unexpected={unexpected} missing={non_backbone}")

    @torch.no_grad()
    def predict(self, audio_path: str) -> dict:
        wav = load_resampled(audio_path, self.cfg.sample_rate, cache_dir=None)
        if len(wav) >= self.max_len:
            start = (len(wav) - self.max_len) // 2
            wav = wav[start:start + self.max_len]
        inputs = self.feature_extractor(
            wav, sampling_rate=self.cfg.sample_rate, return_tensors="pt",
            padding="max_length", max_length=self.max_len, truncation=True,
            return_attention_mask=True,
        )
        logits = self.model(
            inputs.input_values.to(self.device),
            inputs.attention_mask.to(self.device),
        )
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        idx = int(probs.argmax())
        return {
            "emotion": EMOTIONS[idx],
            "confidence": float(probs[idx]),
            "probabilities": {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))},
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict emotion from a speech clip.")
    parser.add_argument("checkpoint")
    parser.add_argument("audio")
    args = parser.parse_args()

    predictor = EmotionPredictor(args.checkpoint)
    result = predictor.predict(args.audio)
    print(f"{result['emotion']}  ({result['confidence']:.1%})")
    for emotion, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {emotion:10s} {p:.1%}")

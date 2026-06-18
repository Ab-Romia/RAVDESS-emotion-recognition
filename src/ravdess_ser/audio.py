"""Audio loading, resampling and batching for the HuBERT pipeline.

RAVDESS ships 48 kHz wav files; HuBERT expects 16 kHz mono. Resampling is the
expensive part, so each clip's 16 kHz waveform is cached to disk the first time
it is read and reused across epochs and folds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

from .config import Config
from .data import Clip


def load_resampled(path: str, target_sr: int, cache_dir: str | None) -> np.ndarray:
    """Load a clip as mono float32 at `target_sr`, caching the result."""
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / (Path(path).stem + f"_{target_sr}.npy")
        if cache_path.exists():
            return np.load(cache_path)

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = soxr.resample(wav, sr, target_sr).astype(np.float32)
    # Trim leading/trailing near-silence so emotion content is centered in the window.
    wav = _trim_silence(wav)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, wav)
    return wav


def _trim_silence(wav: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    """Drop leading/trailing samples below an energy threshold; keep a small margin."""
    if wav.size == 0:
        return wav
    energy = np.abs(wav)
    above = np.where(energy > threshold)[0]
    if above.size == 0:
        return wav
    start = max(0, above[0] - 800)
    end = min(wav.size, above[-1] + 800)
    return wav[start:end]


class RavdessAudioDataset(Dataset):
    def __init__(self, clips: list[Clip], cfg: Config):
        self.clips = clips
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip = self.clips[idx]
        wav = load_resampled(clip.path, self.cfg.sample_rate, self.cfg.cache_dir)
        return wav, clip.emotion


class Collator:
    """Crop to a window and produce encoder-ready batches with dynamic padding.

    Training crops randomly and applies light waveform augmentation (gain, noise,
    time masking) so the model generalizes across the held-out speakers the
    speaker-independent protocol stresses. Validation and test are deterministic:
    center crop, no augmentation.
    """

    def __init__(self, cfg: Config, train: bool = False):
        self.cfg = cfg
        self.train = train and cfg.augment
        self.max_len = int(cfg.sample_rate * cfg.max_seconds)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.ssl_name)

    def __call__(self, batch):
        rng = np.random.default_rng()
        wavs, labels = zip(*batch)
        fitted = [self._prepare(w, rng) for w in wavs]
        inputs = self.feature_extractor(
            fitted,
            sampling_rate=self.cfg.sample_rate,
            return_tensors="pt",
            padding="longest",
            return_attention_mask=True,
        )
        return (
            inputs.input_values,
            inputs.attention_mask,
            torch.tensor(labels, dtype=torch.long),
        )

    def _prepare(self, wav, rng):
        if len(wav) >= self.max_len:
            if self.train:
                start = int(rng.integers(0, len(wav) - self.max_len + 1))
            else:
                start = (len(wav) - self.max_len) // 2
            wav = wav[start:start + self.max_len]
        return self._augment(wav, rng) if self.train else wav

    def _augment(self, wav, rng):
        wav = (wav * rng.uniform(0.85, 1.15)).astype(np.float32)
        if rng.random() < 0.5:
            rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-8)
            snr_db = rng.uniform(15.0, 30.0)
            noise_rms = rms / (10 ** (snr_db / 20))
            wav = wav + rng.normal(0.0, noise_rms, size=wav.shape).astype(np.float32)
        if rng.random() < 0.5 and len(wav) > 1600:
            span = int(rng.integers(int(0.02 * len(wav)), int(0.10 * len(wav)) + 1))
            start = int(rng.integers(0, len(wav) - span))
            wav = wav.copy()
            wav[start:start + span] = 0.0
        return wav.astype(np.float32)

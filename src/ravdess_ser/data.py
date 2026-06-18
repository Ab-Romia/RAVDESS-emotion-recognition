"""Dataset indexing and speaker-independent cross-validation splits.

The single most important thing this module gets right is that no actor ever
appears in both the training and the test side of a fold. RAVDESS has only 24
actors, each speaking the same two sentences, so a naive random split leaks
speaker identity and sentence acoustics into the test set and massively inflates
accuracy. Every split here is built from disjoint groups of actors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import Config, actor_gender


@dataclass(frozen=True)
class Clip:
    path: str
    emotion: int       # 0-7 class index
    actor: int         # 1-24
    gender: str        # "male" / "female"
    intensity: int     # 1 normal, 2 strong
    statement: int     # 1 or 2
    repetition: int    # 1 or 2


def parse_filename(path: Path) -> Clip | None:
    """Parse a RAVDESS filename into a Clip, or return None if it does not match.

    Filename layout: modality-vocal-emotion-intensity-statement-repetition-actor.wav
    e.g. 03-01-06-01-02-01-12.wav
    """
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    try:
        modality, vocal, emotion, intensity, statement, repetition, actor = (
            int(p) for p in parts
        )
    except ValueError:
        return None
    if vocal != 1:  # speech only; ignore song (vocal == 2)
        return None
    if not 1 <= emotion <= 8:
        return None
    return Clip(
        path=str(path),
        emotion=emotion - 1,
        actor=actor,
        gender=actor_gender(actor),
        intensity=intensity,
        statement=statement,
        repetition=repetition,
    )


def index_dataset(data_root: str | Path) -> list[Clip]:
    """Walk the dataset directory and return every speech clip it contains."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    clips = []
    for wav in sorted(root.rglob("*.wav")):
        clip = parse_filename(wav)
        if clip is not None:
            clips.append(clip)
    if not clips:
        raise RuntimeError(f"No RAVDESS speech clips found under {root}")
    return clips


@dataclass(frozen=True)
class Fold:
    index: int
    train_actors: tuple[int, ...]
    val_actors: tuple[int, ...]
    test_actors: tuple[int, ...]


def make_speaker_independent_folds(cfg: Config) -> list[Fold]:
    """Build gender-balanced, actor-disjoint folds.

    Actors are split into `num_folds` test groups of equal size, balanced across
    gender. For each fold the remaining actors form the training pool, from which
    `val_actors_per_fold` actors (also gender-balanced) are held out for early
    stopping. No actor is shared between any of train / val / test within a fold.
    """
    rng = random.Random(cfg.seed)

    males = [a for a in range(1, 25) if actor_gender(a) == "male"]
    females = [a for a in range(1, 25) if actor_gender(a) == "female"]
    rng.shuffle(males)
    rng.shuffle(females)

    if len(males) % cfg.num_folds or len(females) % cfg.num_folds:
        raise ValueError(
            f"{cfg.num_folds} folds does not evenly divide 12 male / 12 female actors"
        )
    m_per = len(males) // cfg.num_folds
    f_per = len(females) // cfg.num_folds

    test_groups = []
    for k in range(cfg.num_folds):
        group = males[k * m_per:(k + 1) * m_per] + females[k * f_per:(k + 1) * f_per]
        test_groups.append(sorted(group))

    folds: list[Fold] = []
    half_val = cfg.val_actors_per_fold // 2
    for k in range(cfg.num_folds):
        test = test_groups[k]
        pool = [a for a in range(1, 25) if a not in test]
        pool_m = [a for a in pool if actor_gender(a) == "male"]
        pool_f = [a for a in pool if actor_gender(a) == "female"]
        # Rotate the validation pick per fold so folds use different val actors,
        # deterministically given the seed.
        val = sorted(_rotate(pool_m, k)[:half_val] + _rotate(pool_f, k)[:half_val])
        train = sorted(a for a in pool if a not in val)
        folds.append(
            Fold(index=k, train_actors=tuple(train),
                 val_actors=tuple(val), test_actors=tuple(test))
        )
    return folds


def _rotate(seq: list[int], k: int) -> list[int]:
    if not seq:
        return seq
    k %= len(seq)
    return seq[k:] + seq[:k]


def clips_for_actors(clips: Iterable[Clip], actors: Iterable[int]) -> list[Clip]:
    actor_set = set(actors)
    return [c for c in clips if c.actor in actor_set]

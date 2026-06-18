"""Guarantees that cross-validation folds are strictly speaker-independent.

These checks are the whole point of the rebuild: if any actor leaks across the
train/test boundary, the reported accuracy is meaningless. Run directly
(`python tests/test_splits.py`) or under pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ravdess_ser.config import Config, actor_gender
from ravdess_ser.data import make_speaker_independent_folds


def test_no_actor_leaks_across_splits():
    folds = make_speaker_independent_folds(Config())
    for f in folds:
        train, val, test = set(f.train_actors), set(f.val_actors), set(f.test_actors)
        assert not (train & test), f"fold {f.index}: actor in train and test"
        assert not (train & val), f"fold {f.index}: actor in train and val"
        assert not (val & test), f"fold {f.index}: actor in val and test"
        assert train | val | test == set(range(1, 25)), \
            f"fold {f.index}: splits must partition all 24 actors"


def test_every_actor_tested_exactly_once():
    folds = make_speaker_independent_folds(Config())
    seen: dict[int, int] = {}
    for f in folds:
        for a in f.test_actors:
            seen[a] = seen.get(a, 0) + 1
    assert sorted(seen) == list(range(1, 25)), "every actor must be tested"
    assert set(seen.values()) == {1}, "each actor must be in exactly one test fold"


def test_test_folds_are_gender_balanced():
    folds = make_speaker_independent_folds(Config())
    for f in folds:
        males = sum(actor_gender(a) == "male" for a in f.test_actors)
        females = sum(actor_gender(a) == "female" for a in f.test_actors)
        assert males == females, f"fold {f.index}: test set not gender balanced"


def test_folds_are_deterministic():
    a = make_speaker_independent_folds(Config(seed=42))
    b = make_speaker_independent_folds(Config(seed=42))
    assert a == b, "folds must be reproducible for a fixed seed"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nAll {len(tests)} split-integrity checks passed.")

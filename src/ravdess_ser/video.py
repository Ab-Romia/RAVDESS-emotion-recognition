"""Face-track extraction from RAVDESS audio-visual clips.

The original deployment center-cropped each frame, which keeps a lot of
background and shifts whenever the actor is off-center. Here a Haar cascade
locates the face per sampled frame and crops to it (with a margin), falling back
to a center crop only when no face is found. Extracted frames are cached.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _CASCADE


def _center_crop(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return frame[y0:y0 + s, x0:x0 + s]


def _face_crop(frame_rgb: np.ndarray, margin: float = 0.35) -> np.ndarray:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                        minSize=(60, 60))
    if len(faces) == 0:
        return _center_crop(frame_rgb)
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    cx, cy = x + w / 2, y + h / 2
    half = max(w, h) * (1 + margin) / 2
    H, W = frame_rgb.shape[:2]
    x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
    y0, y1 = int(max(0, cy - half)), int(min(H, cy + half))
    crop = frame_rgb[y0:y1, x0:x1]
    return crop if crop.size else _center_crop(frame_rgb)


def extract_face_frames(video_path: str, num_frames: int, face_size: int,
                        cache_dir: str | None) -> np.ndarray:
    """Return `num_frames` face crops as a uint8 array (num_frames, S, S, 3)."""
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / (
            Path(video_path).stem + f"_faces{num_frames}x{face_size}.npy")
        if cache_path.exists():
            return np.load(cache_path)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    if total > 0:
        # RAVDESS clips open and close on a neutral face; sample the expressive middle.
        lo, hi = int(0.1 * total), int(0.9 * total)
        for idx in np.linspace(lo, max(lo + 1, hi), num_frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                frames.append(np.zeros((face_size, face_size, 3), np.uint8))
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            crop = cv2.resize(_face_crop(rgb), (face_size, face_size))
            frames.append(crop)
    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1] if frames
                      else np.zeros((face_size, face_size, 3), np.uint8))
    out = np.stack(frames[:num_frames]).astype(np.uint8)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, out)
    return out

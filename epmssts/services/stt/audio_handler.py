from __future__ import annotations

from io import BytesIO
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """
    Convert audio to mono.

    Accepts shapes:
    - (num_samples,)        → mono
    - (num_samples, num_channels) → average across channels
    """
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        # Average channels safely
        return audio.mean(axis=1)
    raise ValueError(f"Unsupported audio shape {audio.shape!r}")


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio to the target sample rate using polyphase filtering.
    """
    if orig_sr == target_sr:
        return audio

    # Use integer ratio resampling via greatest common divisor
    from math import gcd

    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(audio, up, down)


def preprocess_audio_bytes(file_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode arbitrary audio bytes and convert to 16kHz mono float32 PCM.

    Raises:
        ValueError: If the bytes cannot be decoded as audio.
    """
    try:
        audio, sample_rate = sf.read(BytesIO(file_bytes), always_2d=False)
    except Exception as exc:
        raise ValueError(f"Invalid or unsupported audio format: {exc}") from exc

    if audio.size == 0:
        raise ValueError("Decoded audio is empty.")

    audio = _to_mono(np.asarray(audio, dtype=np.float32))
    audio = _resample(audio, int(sample_rate), TARGET_SAMPLE_RATE)

    # Ensure float32 in [-1, 1] as expected by faster-whisper
    # If the source was int PCM, soundfile already normalizes; this clamp
    # is a safe guardrail.
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    return audio, TARGET_SAMPLE_RATE


__all__ = ["preprocess_audio_bytes", "TARGET_SAMPLE_RATE"]


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from faster_whisper import WhisperModel


@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    segments: List[TranscriptionSegment]


class SpeechToTextService:
    """
    Wrapper around faster-whisper for production-ready STT.

    - Loads the `large-v3` model once at process startup.
    - Supports automatic language detection.
    - Accepts preprocessed 16kHz mono float32 NumPy arrays.
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: Optional[str] = None,
        compute_type: str = "float16",
    ) -> None:
        """
        Initialize the Whisper model.

        Args:
            model_size: Model identifier; must be `large-v3` per spec.
            device: "cuda", "cpu", or None for auto-detection.
            compute_type: Precision / quantization mode.
        """
        if model_size != "large-v3":
            # Enforce project spec strictly
            raise ValueError("EPMSSTS requires faster-whisper model 'large-v3'.")

        # Device auto-detection if not explicitly provided
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        # CPU-friendly default if GPU is not available
        if device == "cpu" and compute_type not in {"int8", "int8_float16"}:
            compute_type = "int8"

        self._model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    @staticmethod
    def is_silent(audio: np.ndarray, threshold: float = 1e-4) -> bool:
        """
        Heuristic check for near-silent audio based on RMS energy.
        """
        if audio.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return rms < threshold

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """
        Run transcription on preprocessed audio.

        Args:
            audio: 1D float32 NumPy array at 16kHz mono.
            sample_rate: Sample rate of `audio`. Must be 16_000.
        """
        if audio.ndim != 1:
            raise ValueError("Expected mono audio array of shape (num_samples,).")
        if sample_rate != 16_000:
            raise ValueError("SpeechToTextService expects audio at 16kHz.")

        segments_iter, info = self._model.transcribe(
            audio,
            sample_rate=sample_rate,
            language=None,  # Auto-detect language
            task="transcribe",
        )

        segments: List[TranscriptionSegment] = []
        collected_text: List[str] = []

        for seg in segments_iter:
            text = seg.text.strip()
            segments.append(
                TranscriptionSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=text,
                )
            )
            if text:
                collected_text.append(text)

        full_text = " ".join(collected_text).strip()

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            duration=float(info.duration),
            segments=segments,
        )


__all__ = ["SpeechToTextService", "TranscriptionResult", "TranscriptionSegment"]


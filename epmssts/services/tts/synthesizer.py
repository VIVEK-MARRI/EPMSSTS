from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Literal, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample
import torch

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    TTS = None


SupportedTtsLang = Literal["en", "te", "hi"]
SupportedEmotion = Literal["neutral", "happy", "sad", "angry", "fearful"]


EMOTION_SPEED: Dict[SupportedEmotion, float] = {
    "happy": 1.05,
    "sad": 0.92,
    "angry": 1.10,
    "neutral": 1.00,
    "fearful": 0.95,
}


@dataclass
class TtsSynthesisRequest:
    text: str
    language: SupportedTtsLang
    emotion: SupportedEmotion


class TtsService:
    """
    Emotion-conditioned TTS using Coqui TTS (YourTTS/VITS style models).

    - Loads the TTS model once at startup.
    - Supports English, Telugu, and Hindi text.
    - Emotion affects ONLY speaking speed (no pitch, no timbre changes).
    - Uses a single default speaker.
    """

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/your_tts",
        device: Optional[str] = None,
    ) -> None:
        if not TTS_AVAILABLE:
            raise RuntimeError(
                "TTS module is not available. Install it using: "
                "pip install TTS (requires Python < 3.13)"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        use_gpu = device == "cuda"

        # Initialize Coqui TTS model once.
        self._tts = TTS(model_name=model_name, gpu=use_gpu)

        # Try to read sample rate from underlying synthesizer, fall back to 22.05 kHz.
        default_sr = 22050
        synthesizer = getattr(self._tts, "synthesizer", None)
        sample_rate = getattr(synthesizer, "output_sample_rate", default_sr)
        self._sample_rate: int = int(sample_rate) if sample_rate else default_sr

    def _validate_request(self, request: TtsSynthesisRequest) -> None:
        if not request.text or not request.text.strip():
            raise ValueError("Text must be a non-empty string.")

        if request.language not in ("en", "te", "hi"):
            raise ValueError("Language must be one of: 'en', 'te', 'hi'.")

        if request.emotion not in EMOTION_SPEED:
            raise ValueError(
                "Emotion must be one of: 'neutral', 'happy', 'sad', 'angry', 'fearful'."
            )

    @staticmethod
    def _apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
        """
        Adjust speaking speed by resampling the waveform.

        - speed > 1.0 → faster (shorter audio).
        - speed < 1.0 → slower (longer audio).
        """
        if speed == 1.0:
            return audio

        n_samples = max(1, int(len(audio) / speed))
        if n_samples == len(audio):
            return audio

        return resample(audio, n_samples).astype(np.float32)

    def synthesize(self, request: TtsSynthesisRequest) -> bytes:
        """
        Synthesize speech audio as WAV bytes for the given request.
        """
        self._validate_request(request)

        # For multilingual YourTTS-style models, we rely on the model's
        # inherent multilingual support. The `language` field is currently
        # used for validation and may be wired into model-specific language
        # controls in future iterations if needed.

        # Generate waveform using a default speaker.
        wav = self._tts.tts(text=request.text)
        audio = np.asarray(wav, dtype=np.float32)

        # Apply emotion-based speed control only.
        speed = EMOTION_SPEED[request.emotion]
        audio = self._apply_speed(audio, speed)

        # Serialize to WAV bytes.
        buf = BytesIO()
        sf.write(buf, audio, samplerate=self._sample_rate, format="WAV")
        return buf.getvalue()


__all__ = ["TtsService", "TtsSynthesisRequest", "SupportedTtsLang", "SupportedEmotion"]


from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import os
import tempfile
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

# pyttsx3 will be imported dynamically in __init__ to avoid module loading issues
PYTTSX3_AVAILABLE = True  # Assume available, check at runtime
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
    Emotion-conditioned TTS using Coqui TTS when available, otherwise
    falls back to Windows SAPI via pyttsx3.

    - Supports English, Telugu, and Hindi text (voice availability depends on OS).
    - Emotion affects speaking speed.
    """

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/your_tts",
        device: Optional[str] = None,
    ) -> None:
        self._engine_kind = "coqui"
        self._tts = None
        self._sample_rate = 22050
        self._pyttsx3 = None
        self._base_rate = 200

        if TTS_AVAILABLE:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            use_gpu = device == "cuda"

            # Initialize Coqui TTS model once.
            self._tts = TTS(model_name=model_name, gpu=use_gpu)

            # Try to read sample rate from underlying synthesizer, fall back to 22.05 kHz.
            default_sr = 22050
            synthesizer = getattr(self._tts, "synthesizer", None)
            sample_rate = getattr(synthesizer, "output_sample_rate", default_sr)
            self._sample_rate = int(sample_rate) if sample_rate else default_sr
            return

        if PYTTSX3_AVAILABLE:
            try:
                import pyttsx3
                self._engine_kind = "pyttsx3"
                self._pyttsx3 = pyttsx3.init()
                try:
                    self._base_rate = int(self._pyttsx3.getProperty("rate") or 200)
                except Exception:
                    self._base_rate = 200
                return
            except ImportError:
                pass  # Fall through to error

        raise RuntimeError(
            "No TTS engine available. Install Coqui TTS (Python < 3.13) "
            "or install pyttsx3 for Windows fallback."
        )

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

    def _select_voice(self, language: SupportedTtsLang) -> None:
        if not self._pyttsx3:
            return

        tokens_by_lang = {
            "en": ["english", "en-", "en_"],
            "hi": ["hindi", "hi-", "hi_"],
            "te": ["telugu", "te-", "te_"],
        }
        tokens = tokens_by_lang.get(language, [])
        try:
            for voice in self._pyttsx3.getProperty("voices"):
                meta = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')} {getattr(voice, 'languages', '')}".lower()
                if any(token in meta for token in tokens):
                    self._pyttsx3.setProperty("voice", voice.id)
                    return
        except Exception:
            return

    def synthesize(self, request: TtsSynthesisRequest) -> bytes:
        """
        Synthesize speech audio as WAV bytes for the given request.
        """
        self._validate_request(request)

        if self._engine_kind == "coqui" and self._tts is not None:
            wav = self._tts.tts(text=request.text)
            audio = np.asarray(wav, dtype=np.float32)

            speed = EMOTION_SPEED[request.emotion]
            audio = self._apply_speed(audio, speed)

            buf = BytesIO()
            sf.write(buf, audio, samplerate=self._sample_rate, format="WAV")
            return buf.getvalue()

        if self._engine_kind == "pyttsx3" and self._pyttsx3 is not None:
            self._select_voice(request.language)
            speed = EMOTION_SPEED[request.emotion]
            rate = max(80, int(self._base_rate * speed))
            try:
                self._pyttsx3.setProperty("rate", rate)
            except Exception:
                pass

            tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp_file.name
            tmp_file.close()

            try:
                self._pyttsx3.save_to_file(request.text, tmp_path)
                self._pyttsx3.runAndWait()
                with open(tmp_path, "rb") as handle:
                    return handle.read()
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        raise RuntimeError("TTS engine is not initialized.")


__all__ = ["TtsService", "TtsSynthesisRequest", "SupportedTtsLang", "SupportedEmotion"]


from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import logging
import os
import sys
import tempfile
import time
from typing import Dict, Literal, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample
import torch

logger = logging.getLogger(__name__)

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
    Emotion-conditioned TTS using 3-tier fallback strategy:
    1. Coqui TTS (preferred, multilingual)
    2. pyttsx3 (Windows SAPI fallback)
    3. Synthetic tone generation (always works)

    - Supports English, Telugu, and Hindi text (voice availability depends on OS).
    - Emotion affects speaking speed only (safe, non-destructive).
    - Gracefully degrades to next tier if current engine fails.
    """

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/your_tts",
        device: Optional[str] = None,
    ) -> None:
        self._engine_kind = "uninitialized"
        self._tts = None
        self._sample_rate = 22050
        self._pyttsx3 = None
        self._base_rate = 200

        engine_pref = os.environ.get("EPMSSTS_TTS_ENGINE", "auto").strip().lower()
        allow_coqui = engine_pref in {"auto", "coqui"}
        allow_pyttsx3 = engine_pref in {"auto", "pyttsx3"}
        allow_fallback = engine_pref in {"auto", "fallback"}

        # Coqui TTS is not reliable on Python 3.13; skip unless explicitly forced.
        if sys.version_info >= (3, 13) and engine_pref != "coqui":
            allow_coqui = False
            logger.info("Python 3.13+ detected: Coqui TTS disabled (use pyttsx3 instead)")

        # Tier 1: Try Coqui TTS
        if TTS_AVAILABLE and allow_coqui:
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"

            use_gpu = device == "cuda"

            try:
                logger.info(f"Initializing Coqui TTS (device={device})...")
                # Initialize Coqui TTS model once.
                self._tts = TTS(model_name=model_name, gpu=use_gpu)

                # Try to read sample rate from underlying synthesizer, fall back to 22.05 kHz.
                default_sr = 22050
                synthesizer = getattr(self._tts, "synthesizer", None)
                sample_rate = getattr(synthesizer, "output_sample_rate", default_sr)
                self._sample_rate = int(sample_rate) if sample_rate else default_sr
                
                self._engine_kind = "coqui"
                logger.info(f"✓ Coqui TTS initialized (sample_rate={self._sample_rate})")
                return
            except Exception as exc:
                logger.warning(f"Coqui TTS initialization failed: {exc}")
                self._tts = None

        # Tier 2: Try pyttsx3
        if allow_pyttsx3:
            try:
                logger.info("Initializing pyttsx3...")
                import pyttsx3
                self._pyttsx3 = pyttsx3.init()
                
                try:
                    self._base_rate = int(self._pyttsx3.getProperty("rate") or 200)
                except Exception:
                    self._base_rate = 200
                
                self._engine_kind = "pyttsx3"
                logger.info(f"✓ pyttsx3 initialized (base_rate={self._base_rate})")
                return
            except (ImportError, Exception) as exc:
                logger.warning(f"pyttsx3 initialization failed: {exc}")
                self._pyttsx3 = None

        # Tier 3: Use synthetic fallback
        if allow_fallback:
            self._engine_kind = "fallback"
            logger.info("✓ Using synthetic tone fallback (always works)")
            return

        # No engines available
        raise RuntimeError(
            f"No TTS engine available. "
            f"Install Coqui TTS (Python <3.13) or pyttsx3. "
            f"Current Python: {sys.version_info.major}.{sys.version_info.minor}"
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

        Args:
            audio: Audio waveform (numpy array)
            speed: Speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)

        Returns:
            Resampled audio array
        """
        if speed == 1.0 or len(audio) == 0:
            return audio

        n_samples = max(1, int(len(audio) / speed))
        if n_samples == len(audio):
            return audio

        return resample(audio, n_samples).astype(np.float32)

    def _select_voice(self, language: SupportedTtsLang) -> None:
        """Select appropriate voice for target language (pyttsx3 only)."""
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
                    logger.debug(f"Selected voice for {language}: {voice.id}")
                    return
        except Exception as exc:
            logger.debug(f"Could not select voice: {exc}")

    def synthesize(self, request: TtsSynthesisRequest) -> bytes:
        """
        Synthesize speech audio as WAV bytes.
        
        Uses 3-tier strategy: Coqui TTS → pyttsx3 → synthetic fallback
        
        Args:
            request: TtsSynthesisRequest with text, language, emotion
            
        Returns:
            WAV audio bytes
        """
        self._validate_request(request)

        # Tier 1: Coqui TTS
        if self._engine_kind == "coqui" and self._tts is not None:
            try:
                wav = self._tts.tts(text=request.text)
                audio = np.asarray(wav, dtype=np.float32)

                speed = EMOTION_SPEED[request.emotion]
                audio = self._apply_speed(audio, speed)

                buf = BytesIO()
                sf.write(buf, audio, samplerate=self._sample_rate, format="WAV")
                return buf.getvalue()
            except Exception as exc:
                logger.error(f"Coqui TTS synthesis failed: {exc}", exc_info=True)
                logger.info("Falling back to fallback synthesizer...")

        # Tier 2: pyttsx3
        if self._engine_kind == "pyttsx3" and self._pyttsx3 is not None:
            tmp_path = None
            try:
                self._select_voice(request.language)
                speed = EMOTION_SPEED[request.emotion]
                rate = max(80, int(self._base_rate * speed))
                
                try:
                    self._pyttsx3.setProperty("rate", rate)
                except Exception as exc:
                    logger.warning(f"Could not set speech rate: {exc}")

                # Create temporary file for pyttsx3 output
                tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp_file.name
                tmp_file.close()

                self._pyttsx3.save_to_file(request.text, tmp_path)
                self._pyttsx3.runAndWait()
                
                # Wait for file to be written with content
                wait_count = 0
                while wait_count < 20:  # Max 2 seconds
                    try:
                        file_size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
                        if file_size > 44:  # WAV header minimum
                            break
                    except OSError:
                        pass
                    time.sleep(0.1)
                    wait_count += 1
                
                if not os.path.exists(tmp_path):
                    raise RuntimeError(f"pyttsx3 failed to create output file at {tmp_path}")
                
                file_size = os.path.getsize(tmp_path)
                if file_size <= 44:
                    raise RuntimeError(f"pyttsx3 generated empty audio ({file_size} bytes)")
                
                # Read and return the generated audio
                with open(tmp_path, "rb") as handle:
                    audio_bytes = handle.read()
                    logger.debug(f"pyttsx3 generated {len(audio_bytes)} bytes")
                    return audio_bytes
                    
            except Exception as exc:
                logger.error(f"pyttsx3 synthesis failed: {exc}", exc_info=True)
                logger.info("Falling back to synthetic tone...")
            finally:
                # Clean up temporary file
                if tmp_path:
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception as exc:
                        logger.debug(f"Could not delete temp file: {exc}")

        # Tier 3: Synthetic fallback (always works)
        logger.info("Using synthetic tone fallback")
        return self._synthesize_fallback(request)

    def _synthesize_fallback(self, request: TtsSynthesisRequest) -> bytes:
        """
        Generate synthetic tone-based WAV as last-resort fallback.
        
        This tier always succeeds and guarantees audio output.
        Useful for testing and when TTS engines are unavailable.
        
        Args:
            request: TtsSynthesisRequest
            
        Returns:
            WAV audio bytes (always succeeds)
        """
        try:
            # Duration based on text length
            duration = max(0.6, min(3.0, 0.06 * max(1, len(request.text.split())) + 0.4))
            sr = int(self._sample_rate or 22050)
            
            # Generate time array
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            
            # Base frequency depends on text (deterministic)
            base_freq = 220 + (abs(hash(request.text)) % 120)
            
            # Generate sine wave
            audio = 0.2 * np.sin(2 * np.pi * base_freq * t).astype(np.float32)

            # Apply emotion-based speed
            speed = EMOTION_SPEED[request.emotion]
            audio = self._apply_speed(audio, speed)

            # Encode as WAV
            buf = BytesIO()
            sf.write(buf, audio, samplerate=sr, format="WAV")
            
            audio_bytes = buf.getvalue()
            logger.debug(f"Synthetic tone generated: {len(audio_bytes)} bytes @ {sr}Hz")
            return audio_bytes
            
        except Exception as exc:
            logger.error(f"Fallback synthesis failed (this should never happen): {exc}")
            # Ultimate fallback: return empty WAV header
            return b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'


__all__ = ["TtsService", "TtsSynthesisRequest", "SupportedTtsLang", "SupportedEmotion"]


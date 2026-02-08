from __future__ import annotations

"""
End-to-end speech-to-speech orchestration for EPMSSTS.

This module wires together the existing services:
- STT (faster-whisper)
- Audio-based emotion detection
- Rule-based dialect detection
- Pure text translation
- Emotion-conditioned TTS

Service implementations themselves must not be modified.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Optional, Tuple

import asyncio
from pathlib import Path
from uuid import uuid4

import numpy as np

from epmssts.services.stt.audio_handler import preprocess_audio_bytes
from epmssts.services.stt.transcriber import SpeechToTextService, TranscriptionResult
from epmssts.services.emotion.audio_emotion import AudioEmotionService, EmotionPrediction
from epmssts.services.emotion.text_emotion import TextEmotionService
from epmssts.services.emotion.fusion import fuse_emotions
from epmssts.services.dialect.classifier import DialectClassifier, DialectPrediction
from epmssts.services.translation.translator import TranslationService, TranslationResult
from epmssts.services.tts.synthesizer import TtsService, TtsSynthesisRequest


TargetLang = Literal["en", "te", "hi"]


@dataclass
class SpeechToSpeechResult:
    session_id: str
    transcript: str
    detected_language: str
    detected_emotion: str
    emotion_confidence: float
    detected_dialect: str
    translated_text: str
    audio_path: Path
    latency_ms: int


async def _run_stt_and_emotion(
    audio_16k: np.ndarray,
    sample_rate: int,
    stt_service: SpeechToTextService,
    emotion_service: AudioEmotionService,
) -> Tuple[TranscriptionResult, EmotionPrediction]:
    loop = asyncio.get_event_loop()

    async def _stt() -> TranscriptionResult:
        return await loop.run_in_executor(
            None, stt_service.transcribe, audio_16k, sample_rate
        )

    async def _emotion() -> EmotionPrediction:
        return await loop.run_in_executor(
            None, emotion_service.predict, audio_16k, sample_rate
        )

    return await asyncio.gather(_stt(), _emotion())


async def run_speech_to_speech(
    file_bytes: bytes,
    target_lang: TargetLang,
    *,
    stt_service: SpeechToTextService,
    emotion_service: AudioEmotionService,
    dialect_classifier: DialectClassifier,
    translation_service: TranslationService,
    tts_service: Optional[TtsService],
    text_emotion_service: Optional[TextEmotionService] = None,
    outputs_dir: Path,
    timeout_seconds: float = 120.0,
) -> SpeechToSpeechResult:
    """
    Run the complete speech-to-speech pipeline.

    The entire orchestration is wrapped in an outer timeout in the API
    layer; this function focuses on chaining the existing services.
    """
    start = perf_counter()
    session_id = str(uuid4())

    # Ensure outputs directory exists.
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Audio preprocessing (16kHz mono).
    try:
        audio_16k, sample_rate = preprocess_audio_bytes(file_bytes)
    except ValueError as exc:
        raise ValueError(f"Invalid audio: {exc}") from exc
    except Exception as exc:  # Unexpected decode errors
        raise RuntimeError(f"Unable to decode or preprocess audio: {exc}") from exc

    # Silence handling: short-circuit with neutral emotion and empty output.
    if stt_service.is_silent(audio_16k):
        audio_path = outputs_dir / f"{session_id}.wav"
        audio_path.touch()
        return SpeechToSpeechResult(
            session_id=session_id,
            transcript="",
            detected_language="en",
            detected_emotion="neutral",
            emotion_confidence=1.0,
            detected_dialect="standard_telugu",
            translated_text="",
            audio_path=audio_path,
            latency_ms=int((perf_counter() - start) * 1000),
        )

    # 2) STT + Emotion in parallel.
    try:
        stt_result, emotion_result = await asyncio.wait_for(
            _run_stt_and_emotion(audio_16k, sample_rate, stt_service, emotion_service),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError("STT + emotion phase exceeded time limit") from exc

    transcript = stt_result.text or ""
    detected_language = (stt_result.language or "").lower()
    if detected_language not in {"en", "te", "hi"}:
        # Fallback to English if Whisper returns an unsupported code.
        detected_language = "en"

    detected_emotion = emotion_result.label

    # Optional text emotion for English, with fusion.
    if (
        text_emotion_service is not None
        and detected_language == "en"
        and transcript.strip()
    ):
        loop = asyncio.get_event_loop()

        async def _text_emotion() -> EmotionPrediction:
            return await loop.run_in_executor(
                None, text_emotion_service.predict, transcript
            )

        try:
            text_pred = await asyncio.wait_for(_text_emotion(), timeout=timeout_seconds)
            fused = fuse_emotions(emotion_result, text_pred)
            detected_emotion = fused.label
            emotion_result = fused
        except asyncio.TimeoutError:
            # Keep audio emotion if text emotion is too slow
            pass
        except Exception:
            # Keep audio emotion if text emotion fails
            pass

    # 3) Dialect detection (Telugu only, metadata only).
    if detected_language == "te" and transcript.strip():
        dialect_prediction: DialectPrediction = dialect_classifier.detect(transcript)
        detected_dialect = dialect_prediction.dialect
    else:
        detected_dialect = "standard_telugu"

    # 4) Translation (pure text).
    if not transcript.strip():
        translated_text = ""
    elif detected_language == target_lang:
        translated_text = transcript
    else:
        loop = asyncio.get_event_loop()

        async def _translate() -> TranslationResult:
            return await loop.run_in_executor(
                None,
                translation_service.translate,
                transcript,
                detected_language,
                target_lang,
            )

        try:
            translation_result = await asyncio.wait_for(
                _translate(), timeout=timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError("Translation phase exceeded time limit") from exc

        translated_text = translation_result.translated_text

    # 5) TTS with emotion-conditioned speed (target language).
    if not translated_text.strip():
        # If there is no text to speak, we still complete the session but
        # skip audio synthesis and return an empty (zero-length) WAV file.
        audio_path = outputs_dir / f"{session_id}.wav"
        audio_path.touch()
    elif tts_service is None:
        # TTS is optional in dev environments; attempt a lazy fallback synth.
        try:
            fallback_tts = TtsService()
            tts_request = TtsSynthesisRequest(
                text=translated_text,
                language=target_lang,
                emotion=detected_emotion,
            )
            wav_bytes = fallback_tts.synthesize(tts_request)
            audio_path = outputs_dir / f"{session_id}.wav"
            audio_path.write_bytes(wav_bytes)
        except Exception:
            audio_path = outputs_dir / f"{session_id}.wav"
            audio_path.touch()
    else:
        tts_request = TtsSynthesisRequest(
            text=translated_text,
            language=target_lang,
            emotion=detected_emotion,
        )

        loop = asyncio.get_event_loop()

        async def _tts() -> bytes:
            return await loop.run_in_executor(None, tts_service.synthesize, tts_request)

        try:
            wav_bytes = await asyncio.wait_for(_tts(), timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("TTS phase exceeded time limit") from exc

        audio_path = outputs_dir / f"{session_id}.wav"
        audio_path.write_bytes(wav_bytes)

    latency_ms = int((perf_counter() - start) * 1000)

    return SpeechToSpeechResult(
        session_id=session_id,
        transcript=transcript,
        detected_language=detected_language,
        detected_emotion=detected_emotion,
        emotion_confidence=emotion_result.confidence,
        detected_dialect=detected_dialect,
        translated_text=translated_text,
        audio_path=audio_path,
        latency_ms=latency_ms,
    )


__all__ = ["run_speech_to_speech", "SpeechToSpeechResult", "TargetLang"]


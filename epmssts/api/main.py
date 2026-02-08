from contextlib import asynccontextmanager
import logging
from typing import Optional
from uuid import uuid4
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, status, Form, Request, Body
from fastapi.responses import JSONResponse, Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from epmssts.services.stt.transcriber import SpeechToTextService, TranscriptionResult
from epmssts.services.stt.audio_handler import preprocess_audio_bytes
from epmssts.services.emotion.audio_emotion import AudioEmotionService, EmotionPrediction
from epmssts.services.emotion.text_emotion import TextEmotionService
from epmssts.services.dialect.classifier import DialectClassifier, DialectPrediction
from epmssts.services.translation.translator import (
    TranslationService,
    TranslationResult,
)
from epmssts.services.tts.synthesizer import TtsService, TtsSynthesisRequest
from epmssts.api.pipeline import run_speech_to_speech, SpeechToSpeechResult


logger = logging.getLogger("epmssts.api")

stt_service: Optional[SpeechToTextService] = None
emotion_service: Optional[AudioEmotionService] = None
text_emotion_service: Optional[TextEmotionService] = None
dialect_classifier: Optional[DialectClassifier] = None
translation_service: Optional[TranslationService] = None
tts_service: Optional[TtsService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Ensures the Whisper model is loaded once at startup
    and released cleanly on shutdown.
    """
    global stt_service, emotion_service, text_emotion_service, dialect_classifier, translation_service, tts_service
    try:
        stt_service = SpeechToTextService()
        emotion_service = AudioEmotionService()
        try:
            text_emotion_service = TextEmotionService()
        except Exception as exc:
            logger.warning("Text emotion service unavailable: %s", exc)
            text_emotion_service = None
        dialect_classifier = DialectClassifier()
        translation_service = TranslationService()
        try:
            tts_service = TtsService()
        except RuntimeError as e:
            logger.warning("TTS service not available: %s", e)
            tts_service = None
    except Exception as exc:  # pragma: no cover - startup failure path
        # Fail fast if the model cannot be loaded
        raise RuntimeError(f"Failed to initialize core services: {exc}") from exc

    yield

    # Teardown hook if we ever need explicit cleanup
    stt_service = None
    emotion_service = None
    text_emotion_service = None
    dialect_classifier = None
    translation_service = None
    tts_service = None


app = FastAPI(title="EPMSSTS API", version="0.1.0", lifespan=lifespan)

# Add CORS middleware to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """
    Basic health check endpoint for readiness probes.
    """
    # Only STT is critical; other services can fail gracefully
    if stt_service is None or emotion_service is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "detail": "Critical services not initialized (STT or Emotion)",
            },
        )
    return {
        "status": "ok",
        "stt_available": stt_service is not None,
        "emotion_available": emotion_service is not None,
        "text_emotion_available": text_emotion_service is not None,
        "dialect_available": dialect_classifier is not None,
        "translation_available": translation_service is not None and translation_service._model is not None,
        "tts_available": tts_service is not None
    }


@app.post("/stt/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
):
    """
    Transcribe an uploaded audio file using faster-whisper.

    - Ensures audio is converted to 16kHz mono.
    - Uses language auto-detection.
    - Times out and aborts after 10 seconds.
    """
    if stt_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="STT service is not available",
        )

    # Basic content-type validation; we keep this permissive and rely on
    # decoding errors for final validation.
    if not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type '{file.content_type}'. Expected audio/*.",
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        ) from exc

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        audio_16k, sample_rate = preprocess_audio_bytes(file_bytes)
    except ValueError as exc:
        # Explicit invalid audio errors (e.g. cannot decode)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        # Unexpected preprocessing errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to decode or preprocess audio: {exc}",
        ) from exc

    # Silence / near-silence check
    if stt_service.is_silent(audio_16k):
        # In Phase 1 we only handle STT. Silence-specific emotion handling
        # is deferred to the emotion module.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio appears to be silent or too quiet for transcription.",
        )

    async def _run_transcription() -> TranscriptionResult:
        return await asyncio.get_event_loop().run_in_executor(
            None, stt_service.transcribe, audio_16k, sample_rate
        )

    try:
        result = await asyncio.wait_for(_run_transcription(), timeout=60.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Transcription exceeded 60s timeout limit.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {exc}",
        ) from exc

    return {
        "text": result.text,
        "language": result.language,
        "duration": result.duration,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            for seg in result.segments
        ],
    }


@app.post("/emotion/detect")
async def detect_emotion(
    file: UploadFile = File(..., description="Audio file to analyze for emotion"),
):
    """
    Detect emotion from an uploaded audio file.

    - Reuses 16kHz mono preprocessing.
    - Uses Wav2Vec2-based SER model (audio-only).
    - Returns dominant emotion and per-emotion confidence scores.
    """
    if emotion_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Emotion service is not available",
        )

    if not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type '{file.content_type}'. Expected audio/*.",
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        ) from exc

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        audio_16k, sample_rate = preprocess_audio_bytes(file_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to decode or preprocess audio: {exc}",
        ) from exc

    # Silence handling: return neutral directly (Phase 2 requirement).
    if emotion_service.is_silent(audio_16k):
        neutral_scores = {
            "neutral": 1.0,
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "fearful": 0.0,
        }
        return {
            "emotion": "neutral",
            "confidence": 1.0,
            "scores": neutral_scores,
        }

    async def _run_emotion() -> EmotionPrediction:
        return await asyncio.get_event_loop().run_in_executor(
            None, emotion_service.predict, audio_16k, sample_rate
        )

    try:
        result = await asyncio.wait_for(_run_emotion(), timeout=60.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Emotion detection exceeded 60s timeout limit.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Emotion detection failed: {exc}",
        ) from exc

    return {
        "emotion": result.label,
        "confidence": result.confidence,
        "scores": result.scores,
    }


@app.post("/dialect/detect")
async def detect_dialect(
    request: Request,
    transcript: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None, description="Optional audio file for dialect detection"),
):
    """
    Detect Telugu dialect from a transcript string.

    - Rule-based, text-only heuristics.
    - Dialects: telangana, andhra, standard_telugu.
    - This is METADATA ONLY and must not affect translation.
    """
    if dialect_classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dialect classifier is not available",
        )

    if transcript is None:
        transcript = request.query_params.get("transcript")

    if transcript is None and file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide either 'transcript' or an audio 'file'.",
        )

    resolved_transcript = transcript or ""

    if file is not None:
        if stt_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="STT service is required for audio-based dialect detection",
            )

        if not file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid content type '{file.content_type}'. Expected audio/*.",
            )

        try:
            file_bytes = await file.read()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to read uploaded file: {exc}",
            ) from exc

        if not file_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        try:
            audio_16k, sample_rate = preprocess_audio_bytes(file_bytes)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to decode or preprocess audio: {exc}",
            ) from exc

        if stt_service.is_silent(audio_16k):
            return {
                "dialect": "standard_telugu",
                "confidence": 0.5,
            }

        async def _run_transcription() -> TranscriptionResult:
            return await asyncio.get_event_loop().run_in_executor(
                None, stt_service.transcribe, audio_16k, sample_rate
            )

        try:
            stt_result = await asyncio.wait_for(_run_transcription(), timeout=60.0)
            resolved_transcript = stt_result.text
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Transcription exceeded 10s timeout limit.",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {exc}",
            ) from exc

    # The endpoint assumes the transcript is Telugu; callers should only use
    # this for Telugu text. For non-Telugu text, the classifier will typically
    # fall back to `standard_telugu` with modest confidence.
    prediction: DialectPrediction = dialect_classifier.detect(resolved_transcript)
    return {
        "dialect": prediction.dialect,
        "confidence": prediction.confidence,
    }


@app.post("/translate")
async def translate(
    payload: dict = Body(...),
):
    """
    Pure text translation endpoint using NLLB-200.

    Input JSON:
    {
      "text": str,
      "source_lang": "te|hi|en",
      "target_lang": "te|hi|en"
    }

    Output JSON:
    {
      "translated_text": str,
      "model": "nllb-200-distilled-600M"
    }
    """
    if translation_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Translation service is not available",
        )

    text = payload.get("text")
    source_lang = payload.get("source_lang")
    target_lang = payload.get("target_lang")

    if text is None or not isinstance(text, str) or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'text' must be a non-empty string.",
        )

    if source_lang not in {"te", "hi", "en"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'source_lang' must be one of: 'te', 'hi', 'en'.",
        )

    if target_lang not in {"te", "hi", "en"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'target_lang' must be one of: 'te', 'hi', 'en'.",
        )

    try:
        result: TranslationResult = translation_service.translate(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Translation failed: {exc}",
        ) from exc

    return {
        "translated_text": result.translated_text,
        "model": result.model,
    }


@app.post("/tts/synthesize")
async def synthesize_tts(
    payload: dict = Body(...),
):
    """
    Emotion-conditioned TTS endpoint.

    Input JSON:
    {
      "text": str,
      "language": "en|te|hi",
      "emotion": "neutral|happy|sad|angry|fearful"
    }

    Output:
      Raw WAV audio bytes with content-type `audio/wav`.
    """
    service = tts_service
    if service is None:
        # Lazy fallback init to keep endpoint working even if startup failed.
        try:
            service = TtsService()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"TTS service is not available: {exc}",
            ) from exc

    text = payload.get("text")
    language = payload.get("language")
    emotion = payload.get("emotion")

    if text is None or not isinstance(text, str) or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'text' must be a non-empty string.",
        )

    if language not in {"en", "te", "hi"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'language' must be one of: 'en', 'te', 'hi'.",
        )

    if emotion not in {"neutral", "happy", "sad", "angry", "fearful"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Field 'emotion' must be one of: "
                "'neutral', 'happy', 'sad', 'angry', 'fearful'."
            ),
        )

    request = TtsSynthesisRequest(
        text=text,
        language=language,
        emotion=emotion,
    )

    try:
        wav_bytes = service.synthesize(request)
    except ValueError as exc:
        logger.error(f"TTS validation error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error(f"TTS synthesis failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTS synthesis failed: {exc}",
        ) from exc

    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/translate/speech")
async def translate_speech(
    file: UploadFile = File(..., description="Input audio file for speech-to-speech translation"),
    target_lang: str = Form(..., description="Target language code: en|te|hi"),
):
    """
    End-to-end speech-to-speech translation.

    Chains:
    - STT
    - Audio-based emotion detection
    - Telugu dialect detection (metadata only)
    - Pure text translation
    - Emotion-conditioned TTS
    """
    if (
        stt_service is None
        or emotion_service is None
        or dialect_classifier is None
        or translation_service is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Core services are not available",
        )

    if not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type '{file.content_type}'. Expected audio/*.",
        )

    if target_lang not in {"en", "te", "hi"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'target_lang' must be one of: 'en', 'te', 'hi'.",
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        ) from exc

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    outputs_dir = Path(__file__).resolve().parents[2] / "outputs"

    async def _run_pipeline() -> SpeechToSpeechResult:
        return await run_speech_to_speech(
            file_bytes=file_bytes,
            target_lang=target_lang,  # type: ignore[arg-type]
            stt_service=stt_service,  # type: ignore[arg-type]
            emotion_service=emotion_service,  # type: ignore[arg-type]
            dialect_classifier=dialect_classifier,  # type: ignore[arg-type]
            translation_service=translation_service,  # type: ignore[arg-type]
            tts_service=tts_service,
            text_emotion_service=text_emotion_service,
            outputs_dir=outputs_dir,
        )

    try:
        result = await asyncio.wait_for(_run_pipeline(), timeout=120.0)
    except asyncio.TimeoutError:
        print("[WARN] /translate/speech request timed out after 120s.")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="End-to-end translation exceeded 120s timeout limit.",
        )
    except ValueError as exc:
        print(f"[ERROR] /translate/speech invalid request: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        print(f"[ERROR] /translate/speech failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error during speech-to-speech translation.",
        ) from exc

    audio_url = f"/output/{result.session_id}.wav"

    return {
        "session_id": result.session_id,
        "transcript": result.transcript,
        "detected_language": result.detected_language,
        "detected_emotion": result.detected_emotion,
        "emotion_confidence": result.emotion_confidence,
        "detected_dialect": result.detected_dialect,
        "translated_text": result.translated_text,
        "audio_url": audio_url,
        "latency_ms": result.latency_ms,
    }


@app.get("/output/{session_id}.wav")
async def get_output_audio(session_id: str):
    """
    Serve synthesized audio files by session ID.
    """
    outputs_dir = Path(__file__).resolve().parents[2] / "outputs"
    audio_path = outputs_dir / f"{session_id}.wav"

    if not audio_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found.",
        )

    return FileResponse(path=audio_path, media_type="audio/wav")


# Additional convenience endpoints to match frontend expectations
@app.post("/emotion/analyze")
async def analyze_emotion(
    file: UploadFile = File(..., description="Audio file to analyze for emotion"),
):
    """
    Alias for /emotion/detect endpoint for frontend compatibility.
    """
    return await detect_emotion(file)


@app.post("/process/speech-to-speech")
async def process_speech_to_speech(
    file: UploadFile = File(..., description="Input audio file for speech-to-speech translation"),
    target_lang: str = Form(..., description="Target language code (en, te, hi)"),
    target_emotion: str = Form("neutral", description="Ignored; emotion is derived from input audio"),
):
    """
    Complete end-to-end speech-to-speech translation pipeline.
    
    This endpoint:
    1. Transcribes the audio
    2. Detects emotion and dialect
    3. Translates to target language
    4. Synthesizes speech with target emotion
    
    Returns both translated text and audio output.
    """
    if (
        stt_service is None
        or emotion_service is None
        or dialect_classifier is None
        or translation_service is None
    ):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required services not initialized",
        )

    if target_lang not in {"en", "te", "hi"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'target_lang' must be one of: 'en', 'te', 'hi'.",
        )

    if not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type '{file.content_type}'. Expected audio/*.",
        )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        ) from exc

    if not file_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    outputs_dir = Path(__file__).resolve().parents[2] / "outputs"

    async def _run_pipeline() -> SpeechToSpeechResult:
        return await run_speech_to_speech(
            file_bytes=file_bytes,
            target_lang=target_lang,  # type: ignore[arg-type]
            stt_service=stt_service,  # type: ignore[arg-type]
            emotion_service=emotion_service,  # type: ignore[arg-type]
            dialect_classifier=dialect_classifier,  # type: ignore[arg-type]
            translation_service=translation_service,  # type: ignore[arg-type]
            tts_service=tts_service,
            text_emotion_service=text_emotion_service,
            outputs_dir=outputs_dir,
        )

    try:
        result = await asyncio.wait_for(_run_pipeline(), timeout=15.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="End-to-end translation exceeded 15s timeout limit.",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {exc}",
        ) from exc

    output_audio_url = f"/output/{result.session_id}.wav"

    return {
        "transcript": result.transcript,
        "detected_language": result.detected_language,
        "detected_emotion": result.detected_emotion,
        "detected_dialect": result.detected_dialect,
        "translated_text": result.translated_text,
        "target_language": target_lang,
        "target_emotion": result.detected_emotion,
        "output_audio_url": output_audio_url,
        "confidence": result.emotion_confidence,
    }


__all__ = ["app"]


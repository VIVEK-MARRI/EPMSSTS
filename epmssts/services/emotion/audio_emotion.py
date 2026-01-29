from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


MODEL_ID = "superb/wav2vec2-base-superb-er"


EMOTIONS = ("neutral", "happy", "sad", "angry", "fearful")


@dataclass
class EmotionPrediction:
    label: str
    confidence: float
    scores: Dict[str, float]


class AudioEmotionService:
    """
    Audio-based emotion recognition using a Wav2Vec2 SER model.

    - Uses `superb/wav2vec2-base-superb-er` (4 emotions).
    - Maps model outputs into the project emotion set:
      {neutral, happy, sad, angry, fearful}.
    - Accepts 16kHz mono float32 NumPy arrays.
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = torch.device(device)
        self._extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self._model = AutoModelForAudioClassification.from_pretrained(model_id).to(
            self._device
        )

        # Prepare mapping from model labels to our canonical emotions.
        self._model_id2label: Dict[int, str] = dict(
            self._model.config.id2label
        )  # type: ignore[arg-type]
        self._label2emotion: Dict[str, str] = self._build_label_mapping()

    @staticmethod
    def is_silent(audio: np.ndarray, threshold: float = 1e-4) -> bool:
        if audio.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return rms < threshold

    def _build_label_mapping(self) -> Dict[str, str]:
        """
        Map raw model labels to project emotion categories.

        For `superb/wav2vec2-base-superb-er`, labels are already
        one of: angry, happy, sad, neutral.

        We map them directly and reserve `fearful` for future models
        that might expose it explicitly.
        """
        mapping: Dict[str, str] = {}
        for label in self._model_id2label.values():
            lower = label.lower()
            if lower in {"angry", "happy", "sad", "neutral"}:
                mapping[label] = lower
            else:
                # Any unexpected label falls back to neutral.
                mapping[label] = "neutral"
        return mapping

    def predict(self, audio: np.ndarray, sample_rate: int) -> EmotionPrediction:
        """
        Predict emotion from 16kHz mono audio.

        Args:
            audio: 1D float32 NumPy array (16kHz mono).
            sample_rate: sample rate of `audio`. Must be 16_000.
        """
        if audio.ndim != 1:
            raise ValueError("Expected mono audio array of shape (num_samples,).")
        if sample_rate != 16_000:
            raise ValueError("AudioEmotionService expects audio at 16kHz.")

        if self.is_silent(audio):
            # Silence handling: immediate neutral with full confidence.
            scores = {emotion: 0.0 for emotion in EMOTIONS}
            scores["neutral"] = 1.0
            return EmotionPrediction(label="neutral", confidence=1.0, scores=scores)

        inputs = self._extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        # Convert to probabilities.
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

        # Aggregate probabilities into our canonical emotion set.
        canonical_scores: Dict[str, float] = {e: 0.0 for e in EMOTIONS}
        for idx, prob in enumerate(probs):
            raw_label = self._model_id2label[int(idx)]
            emotion = self._label2emotion.get(raw_label, "neutral")
            canonical_scores[emotion] += float(prob)

        # Normalize to sum to 1.0 to stay well-formed.
        total = float(sum(canonical_scores.values()))
        if total > 0:
            canonical_scores = {k: v / total for k, v in canonical_scores.items()}

        # Choose the top emotion.
        top_label = max(canonical_scores.items(), key=lambda kv: kv[1])[0]
        confidence = canonical_scores[top_label]

        return EmotionPrediction(
            label=top_label,
            confidence=confidence,
            scores=canonical_scores,
        )


__all__ = ["AudioEmotionService", "EmotionPrediction", "EMOTIONS"]


"""
Emotion fusion utilities.

Audio is the primary signal. Text is optional and only used for English.
We use confidence thresholds to avoid overriding strong audio evidence.
"""

from __future__ import annotations

from typing import Dict, Optional

from epmssts.services.emotion.audio_emotion import EMOTIONS, EmotionPrediction


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
	total = float(sum(scores.values()))
	if total <= 0:
		return {e: 0.0 for e in EMOTIONS}
	return {k: v / total for k, v in scores.items()}


def fuse_emotions(
	audio_pred: EmotionPrediction,
	text_pred: Optional[EmotionPrediction],
	*,
	text_min_confidence: float = 0.40,
	audio_min_confidence: float = 0.40,
	audio_weight: float = 0.65,
	text_weight: float = 0.35,
) -> EmotionPrediction:
	"""
	Fuse audio and text emotion predictions.

	Rules:
	- If no text prediction, return audio prediction.
	- If text confidence is low, keep audio.
	- If audio confidence is low but text is confident, prefer text.
	- Otherwise, weighted average of normalized scores.
	"""
	if text_pred is None:
		return audio_pred

	if text_pred.confidence < text_min_confidence:
		return audio_pred

	if audio_pred.confidence < audio_min_confidence and text_pred.confidence >= text_min_confidence:
		return text_pred

	audio_scores = _normalize(audio_pred.scores)
	text_scores = _normalize(text_pred.scores)

	combined: Dict[str, float] = {e: 0.0 for e in EMOTIONS}
	for emotion in EMOTIONS:
		combined[emotion] = (
			audio_weight * audio_scores.get(emotion, 0.0)
			+ text_weight * text_scores.get(emotion, 0.0)
		)

	combined = _normalize(combined)
	top_label = max(combined.items(), key=lambda kv: kv[1])[0]
	confidence = combined[top_label]

	return EmotionPrediction(
		label=top_label,
		confidence=confidence,
		scores=combined,
	)


__all__ = ["fuse_emotions"]


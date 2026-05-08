"""GLiNER-based PII recognizer — zero-shot, BERT-sized, CPU-capable."""
from __future__ import annotations

import logging
import re
from typing import Optional

from presidio_analyzer import EntityRecognizer, RecognizerResult

logger = logging.getLogger(__name__)

# Map from our GLiNER prompt labels → Presidio entity types
GLINER_LABEL_TO_ENTITY: dict[str, str] = {
    "person name": "PERSON",
    "patient name": "PERSON",
    "insured name": "PERSON",
    "employer organization": "ORG",
    "employer organization name": "ORG",
    "social security number": "US_SSN",
    "ssn": "US_SSN",
    "date of birth": "DATE_TIME",
    "phone number": "PHONE_NUMBER",
    "email address": "EMAIL_ADDRESS",
    "insurance policy number": "POLICY_NUM",
    "policy number": "POLICY_NUM",
    "street address": "LOCATION",
    "patient address": "LOCATION",
    "national provider identifier": "NPI",
    "npi": "NPI",
    "employer identification number": "EIN",
    "ein": "EIN",
}

# Entity prompts we ask GLiNER at inference time
GLINER_ENTITY_PROMPTS: list[str] = list(GLINER_LABEL_TO_ENTITY.keys())


_DIGIT_RE = re.compile(r"\d")
_EMAIL_RE = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
_SSN_RE = re.compile(r"\d{3}[-\s]?\d{2}[-\s]?\d{4}")


def _validate_span(span: str, entity_type: str) -> bool:
    """Return False to discard GLiNER hits that fail format sanity checks.

    GLiNER sometimes returns form-label text ("Email:", "Phone") or surrounding
    context instead of the actual value. These guards prevent high-FP entity types
    from degrading precision.
    """
    span = span.strip()
    if not span:
        return False
    if entity_type == "EMAIL_ADDRESS":
        return bool(_EMAIL_RE.search(span))
    if entity_type == "PHONE_NUMBER":
        return bool(_DIGIT_RE.search(span)) and len(re.sub(r"\D", "", span)) >= 7
    if entity_type == "US_SSN":
        return bool(_SSN_RE.search(span))
    if entity_type in ("NPI", "EIN"):
        return bool(_DIGIT_RE.search(span))
    return True


class GLiNERRecognizer(EntityRecognizer):
    """Presidio-compatible wrapper around a GLiNER PII detection model.

    The model is lazy-loaded on first call. CPU-capable; GPU used if available.
    Default model: urchade/gliner_medium-v2.1 (general-purpose; swap for
    nvidia/gliner-PII for healthcare/insurance PHI when available).
    """

    DEFAULT_MODEL = "urchade/gliner_medium-v2.1"
    MIN_SCORE = 0.4  # GLiNER scores tend to be lower than spaCy's 0.85

    def __init__(self, model_id: str | None = None, supported_language: str = "en"):
        self._model_id = model_id or self.DEFAULT_MODEL
        self._model = None
        supported_entities = list(set(GLINER_LABEL_TO_ENTITY.values()))
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name="GLiNERRecognizer",
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from gliner import GLiNER  # noqa: PLC0415

            self._model = GLiNER.from_pretrained(self._model_id)
            logger.info("GLiNER model loaded: %s", self._model_id)
        except Exception as exc:
            logger.error("Failed to load GLiNER model %s: %s", self._model_id, exc)
            raise

    def load(self) -> None:
        """Required by Presidio EntityRecognizer interface — deferred to first analyze()."""
        pass

    def analyze(
        self, text: str, entities: list[str], nlp_artifacts=None
    ) -> list[RecognizerResult]:
        if not text or not text.strip():
            return []

        try:
            self._load()
        except Exception:
            return []

        try:
            raw = self._model.predict_entities(
                text, GLINER_ENTITY_PROMPTS, threshold=self.MIN_SCORE
            )
        except Exception as exc:
            logger.warning("GLiNER predict_entities failed: %s", exc)
            return []

        results = []
        for hit in raw:
            presidio_type = GLINER_LABEL_TO_ENTITY.get(hit["label"].lower())
            if presidio_type is None:
                continue
            if entities and presidio_type not in entities:
                continue
            span = text[hit["start"]:hit["end"]]
            if not _validate_span(span, presidio_type):
                continue
            results.append(
                RecognizerResult(
                    entity_type=presidio_type,
                    start=hit["start"],
                    end=hit["end"],
                    score=min(hit["score"], 1.0),
                )
            )
        return results

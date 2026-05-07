"""Fine-tuned spaCy insurance NER wrapper with graceful degradation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pii_redact.config import settings

logger = logging.getLogger(__name__)


class InsuranceNER:
    """Wraps the fine-tuned insurance NER spaCy model.

    Lazy-loads the model on first call to ``analyze``.  If the model
    directory does not exist, logs a warning and returns an empty list so
    the rest of the pipeline continues unaffected.
    """

    def __init__(self) -> None:
        self._nlp: Any | None = None
        self._load_attempted: bool = False
        self._model_path = Path(settings.models.insurance_ner_model).resolve()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load(self) -> None:
        """Attempt to load the spaCy model exactly once."""
        if self._load_attempted:
            return
        self._load_attempted = True

        if not self._model_path.exists():
            logger.warning(
                "Insurance NER model not found at %s — entity detection disabled "
                "(train the model and place it there to enable).",
                self._model_path,
            )
            return

        try:
            import spacy  # noqa: PLC0415

            self._nlp = spacy.load(str(self._model_path))
            logger.info("Insurance NER model loaded from %s", self._model_path)
        except Exception as exc:
            logger.warning(
                "Failed to load insurance NER model from %s: %s",
                self._model_path,
                exc,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> list[dict[str, Any]]:
        """Run the insurance NER model over *text*.

        Returns a list of dicts::

            [{"text": str, "label": str, "start": int, "end": int}, ...]

        Returns an empty list if the model is unavailable or *text* is blank.
        """
        if not text or not text.strip():
            return []

        self._try_load()

        if self._nlp is None:
            return []

        doc = self._nlp(text)
        results: list[dict[str, Any]] = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]
        return results

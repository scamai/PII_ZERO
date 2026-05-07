"""Build and cache a configured Presidio AnalyzerEngine (singleton)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pii_redact.config import settings

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine

logger = logging.getLogger(__name__)

_analyzer_instance: "AnalyzerEngine | None" = None


def build_analyzer_engine() -> "AnalyzerEngine":
    """Return the singleton AnalyzerEngine, building it on first call.

    spaCy is loaded from the local path specified in settings.models.spacy_model.
    All insurance-domain PatternRecognizers are registered in addition to the
    standard Presidio built-ins.
    """
    global _analyzer_instance
    if _analyzer_instance is not None:
        return _analyzer_instance

    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    from pii_redact.ner.regex_patterns import PRESIDIO_RECOGNIZER_LIST

    spacy_model_path = str(
        Path(settings.models.spacy_model).resolve()
    )

    # Build the NLP engine pointing at a local spaCy model.
    # NlpEngineProvider accepts a "models" list; each entry maps a language to a
    # model name. Because we are offline, the model *name* is the local path.
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model_path}],
    }

    try:
        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        nlp_engine = provider.create_engine()
        logger.info("spaCy NLP engine loaded from %s", spacy_model_path)
    except Exception as exc:
        logger.warning(
            "Could not load spaCy model from %s (%s). "
            "Falling back to blank spaCy model — NLP-based entity recognition disabled.",
            spacy_model_path,
            exc,
        )
        # Graceful degradation: use blank English model so Presidio still works
        # for regex-based recognizers.
        import spacy  # noqa: PLC0415

        spacy.blank("en")  # ensure spaCy is importable
        nlp_configuration_blank = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "blank:en"}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_configuration_blank)
        nlp_engine = provider.create_engine()

    # Build registry and register all custom recognizers.
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)

    for recognizer in PRESIDIO_RECOGNIZER_LIST:
        registry.add_recognizer(recognizer)
        logger.debug("Registered recognizer: %s", recognizer.name)

    _analyzer_instance = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["en"],
    )
    logger.info(
        "AnalyzerEngine ready with %d recognizers",
        len(registry.recognizers),
    )
    return _analyzer_instance

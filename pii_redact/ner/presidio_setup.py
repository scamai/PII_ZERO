"""Build and cache a configured Presidio AnalyzerEngine (singleton)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from pii_redact.config import settings

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine

logger = logging.getLogger(__name__)

_analyzer_instance: "AnalyzerEngine | None" = None


def build_analyzer_engine(use_gliner: bool = False) -> "AnalyzerEngine":
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

    # Resolve model name: prefer the configured local path if it exists on disk,
    # then fall back through progressively smaller installed spaCy models.
    import spacy as _spacy

    configured_path = Path(settings.models.spacy_model).resolve()
    _candidates = [
        str(configured_path),
        "en_core_web_lg",
        "en_core_web_md",
        "en_core_web_sm",
    ]
    spacy_model_path = "blank:en"
    for candidate in _candidates:
        try:
            _spacy.load(candidate)
            spacy_model_path = candidate
            break
        except Exception:
            continue

    logger.info("Using spaCy model: %s", spacy_model_path)

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

    # Remove built-ins replaced by our custom recognizers:
    # - UsBankRecognizer: FPs dominate (P~0.03); our ROUTING_NUM uses context
    # - CreditCardRecognizer: misses 19-digit Maestro; replaced by CreditCard19Recognizer
    for noisy in ("UsBankRecognizer", "CreditCardRecognizer"):
        try:
            registry.remove_recognizer(noisy)
            logger.debug("Removed built-in recognizer: %s", noisy)
        except Exception:
            pass

    for recognizer in PRESIDIO_RECOGNIZER_LIST:
        registry.add_recognizer(recognizer)
        logger.debug("Registered recognizer: %s", recognizer.name)

    # Optionally add GLiNER zero-shot NER layer (opt-in via env var or parameter).
    _use_gliner = use_gliner or os.environ.get("PII_USE_GLINER", "").lower() in ("1", "true", "yes")
    if _use_gliner:
        try:
            from pii_redact.ner.gliner_recognizer import GLiNERRecognizer  # noqa: PLC0415

            gliner = GLiNERRecognizer()
            registry.add_recognizer(gliner)
            logger.info("GLiNER recognizer registered")
        except Exception as exc:
            logger.warning("GLiNER not available, skipping: %s", exc)

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

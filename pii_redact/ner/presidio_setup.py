"""Build and cache Presidio AnalyzerEngine and StructuredEngine singletons.

Environment flags (all opt-in, default off):
  PII_USE_TRANSFORMERS=1   Swap spaCy NLP backend for a HuggingFace NER model.
                           Override model with PII_TRANSFORMERS_MODEL=<hf-repo>.
                           Default model: obi/deid_roberta_i2b2 (clinical de-ID,
                           trained on i2b2 PHI; more relevant than news-corpus models).
                           Falls back to spaCy silently if model not cached.
  PII_USE_GLINER=1         Add GLiNER zero-shot NER layer after spaCy/transformer NER.
  PII_USE_STRUCTURED=1     Expose StructuredEngine for tabular data analysis
                           (bank statement columns, Docling table cells).
"""

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
_structured_engine_instance = None  # presidio_structured.StructuredEngine


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_spacy_model() -> str:
    """Return the best available spaCy model name."""
    import spacy as _spacy

    configured_path = Path(settings.models.spacy_model).resolve()
    candidates = [str(configured_path), "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
    for candidate in candidates:
        try:
            _spacy.load(candidate)
            return candidate
        except Exception:
            continue
    return "blank:en"


def _build_nlp_engine(spacy_model_path: str):
    """Return an NLP engine — TransformersNlpEngine if PII_USE_TRANSFORMERS=1, else spaCy."""
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    use_transformers = os.environ.get("PII_USE_TRANSFORMERS", "").lower() in ("1", "true", "yes")

    if use_transformers:
        hf_model = os.environ.get("PII_TRANSFORMERS_MODEL", "obi/deid_roberta_i2b2")
        try:
            from presidio_analyzer.nlp_engine import TransformersNlpEngine  # noqa: PLC0415

            # TransformersNlpEngine needs spaCy for tokenization + HF model for NER.
            # obi/deid_roberta_i2b2 outputs B-NAME, I-NAME, B-DATE, etc. — clinical PHI labels.
            nlp_engine = TransformersNlpEngine(
                models=[{
                    "lang_code": "en",
                    "model_name": {
                        "spacy": spacy_model_path if spacy_model_path != "blank:en" else "en_core_web_sm",
                        "transformers": hf_model,
                    },
                }]
            )
            nlp_engine.load()
            logger.info("TransformersNlpEngine loaded: %s (spaCy: %s)", hf_model, spacy_model_path)
            return nlp_engine
        except Exception as exc:
            logger.warning(
                "TransformersNlpEngine failed for %s (%s) — falling back to spaCy.",
                hf_model, exc,
            )

    # Default: spaCy NLP engine
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model_path}],
    }
    try:
        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        engine = provider.create_engine()
        logger.info("spaCy NLP engine loaded: %s", spacy_model_path)
        return engine
    except Exception as exc:
        logger.warning(
            "Could not load spaCy model %s (%s) — falling back to blank:en.",
            spacy_model_path, exc,
        )
        blank_cfg = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "blank:en"}],
        }
        provider = NlpEngineProvider(nlp_configuration=blank_cfg)
        return provider.create_engine()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_analyzer_engine(use_gliner: bool = False) -> "AnalyzerEngine":
    """Return the singleton AnalyzerEngine, building it on first call."""
    global _analyzer_instance
    if _analyzer_instance is not None:
        return _analyzer_instance

    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from pii_redact.ner.regex_patterns import PRESIDIO_RECOGNIZER_LIST

    spacy_model_path = _resolve_spacy_model()
    logger.info("Using spaCy model: %s", spacy_model_path)

    nlp_engine = _build_nlp_engine(spacy_model_path)

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)

    # Remove built-ins replaced by domain-tuned custom recognizers:
    # - UsBankRecognizer: FPs dominate (P~0.03); our ABA pattern uses context gate
    # - CreditCardRecognizer: misses 19-digit Maestro; replaced by CreditCard19Recognizer+Luhn
    # - PhoneRecognizer: base score 0.4 misses bare numbers; ours scores 0.65
    # - UsItinRecognizer: base score 0.5 below threshold; replaced by ITIN_Recognizer (0.65)
    # - UsPassportRecognizer: base score 0.45 below threshold; replaced by Passport_Recognizer (0.65)
    for noisy in (
        "UsBankRecognizer",
        "CreditCardRecognizer",
        "PhoneRecognizer",
        "UsItinRecognizer",
        "UsPassportRecognizer",
    ):
        try:
            registry.remove_recognizer(noisy)
            logger.debug("Removed built-in recognizer: %s", noisy)
        except Exception:
            pass

    for recognizer in PRESIDIO_RECOGNIZER_LIST:
        registry.add_recognizer(recognizer)
        logger.debug("Registered recognizer: %s", recognizer.name)

    _use_gliner = use_gliner or os.environ.get("PII_USE_GLINER", "").lower() in ("1", "true", "yes")
    if _use_gliner:
        try:
            from pii_redact.ner.gliner_recognizer import GLiNERRecognizer  # noqa: PLC0415

            registry.add_recognizer(GLiNERRecognizer())
            logger.info("GLiNER recognizer registered")
        except Exception as exc:
            logger.warning("GLiNER not available, skipping: %s", exc)

    _analyzer_instance = AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=["en"],
    )
    logger.info("AnalyzerEngine ready with %d recognizers", len(registry.recognizers))
    return _analyzer_instance


def build_structured_engine():
    """Return a StructuredEngine wrapping the singleton AnalyzerEngine.

    Used for tabular PII analysis: bank statement columns, Docling table cells.
    Requires presidio-structured (pip install presidio-structured).

    Usage::

        engine = build_structured_engine()
        analysis = PandasAnalysisBuilder().generate_analysis(df)
        results = engine.analyze(df, analysis)
        redacted = engine.anonymize(df, results, operators={"DEFAULT": OperatorConfig("replace")})
    """
    global _structured_engine_instance
    if _structured_engine_instance is not None:
        return _structured_engine_instance

    try:
        from presidio_structured import StructuredEngine  # noqa: PLC0415

        analyzer = build_analyzer_engine()
        _structured_engine_instance = StructuredEngine(data_processor=None)
        # Attach the analyzer so StructuredEngine uses our custom recognizer set.
        _structured_engine_instance.analyzer_engine = analyzer
        logger.info("StructuredEngine ready")
    except ImportError:
        logger.warning(
            "presidio-structured not installed — structured tabular analysis unavailable. "
            "Install with: pip install presidio-structured"
        )
    except Exception as exc:
        logger.warning("StructuredEngine failed to initialize: %s", exc)

    return _structured_engine_instance


def reset_engines() -> None:
    """Clear cached engine singletons (for testing)."""
    global _analyzer_instance, _structured_engine_instance
    _analyzer_instance = None
    _structured_engine_instance = None

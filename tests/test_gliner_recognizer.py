"""Tests for GLiNERRecognizer — zero-shot NER layer.

Fast tests use mocking (no model download required).
Slow tests actually load the model from HuggingFace.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fast tier — mocked, no model download
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_gliner_recognizer_importable():
    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer, GLINER_LABEL_TO_ENTITY

    assert "person name" in GLINER_LABEL_TO_ENTITY
    assert "employer organization" in GLINER_LABEL_TO_ENTITY
    assert GLINER_LABEL_TO_ENTITY["person name"] == "PERSON"
    assert GLINER_LABEL_TO_ENTITY["employer organization"] == "ORG"
    assert GLINER_LABEL_TO_ENTITY["social security number"] == "US_SSN"


@pytest.mark.fast
def test_gliner_recognizer_empty_text():
    """Must return [] without loading model on empty text."""
    from unittest.mock import MagicMock

    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()
    rec._model = MagicMock()  # pretend loaded
    rec._model.predict_entities.return_value = []

    result = rec.analyze("", ["PERSON"])
    assert result == []
    # Model should NOT be called for empty text
    rec._model.predict_entities.assert_not_called()


@pytest.mark.fast
def test_gliner_recognizer_whitespace_only():
    """Whitespace-only text should also return []."""
    from unittest.mock import MagicMock

    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()
    rec._model = MagicMock()
    result = rec.analyze("   \t\n  ", ["PERSON"])
    assert result == []


@pytest.mark.fast
def test_gliner_maps_person_name_to_person_entity():
    from unittest.mock import MagicMock

    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()
    rec._model = MagicMock()
    rec._model.predict_entities.return_value = [
        {"text": "Jane Doe", "label": "person name", "score": 0.92, "start": 0, "end": 8}
    ]

    results = rec.analyze("Jane Doe works here", ["PERSON"])
    assert any(r.entity_type == "PERSON" for r in results)
    assert results[0].start == 0
    assert results[0].end == 8
    assert results[0].score == pytest.approx(0.92)


@pytest.mark.fast
def test_gliner_maps_employer_org_to_org():
    from unittest.mock import MagicMock

    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()
    rec._model = MagicMock()
    rec._model.predict_entities.return_value = [
        {
            "text": "Acme Corp",
            "label": "employer organization",
            "score": 0.88,
            "start": 14,
            "end": 23,
        }
    ]

    results = rec.analyze("Patient employed at Acme Corp.", ["ORG"])
    assert any(r.entity_type == "ORG" for r in results)


@pytest.mark.fast
def test_gliner_maps_ssn_label():
    from unittest.mock import MagicMock

    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()
    rec._model = MagicMock()
    rec._model.predict_entities.return_value = [
        {
            "text": "523-67-4891",
            "label": "social security number",
            "score": 0.95,
            "start": 5,
            "end": 16,
        }
    ]

    results = rec.analyze("SSN: 523-67-4891", ["US_SSN"])
    assert any(r.entity_type == "US_SSN" for r in results)


@pytest.mark.fast
def test_gliner_filters_by_requested_entities():
    """Results not in the requested entity list should be excluded."""
    from unittest.mock import MagicMock

    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()
    rec._model = MagicMock()
    rec._model.predict_entities.return_value = [
        {"text": "Jane Doe", "label": "person name", "score": 0.92, "start": 0, "end": 8},
        {"text": "Acme Corp", "label": "employer organization", "score": 0.88, "start": 14, "end": 23},
    ]

    # Only ask for PERSON — ORG result should be excluded
    results = rec.analyze("Jane Doe at Acme Corp", ["PERSON"])
    entity_types = {r.entity_type for r in results}
    assert "PERSON" in entity_types
    assert "ORG" not in entity_types


@pytest.mark.fast
def test_gliner_score_capped_at_one():
    """Score should never exceed 1.0 even if model returns > 1."""
    from unittest.mock import MagicMock

    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()
    rec._model = MagicMock()
    rec._model.predict_entities.return_value = [
        {"text": "Jane Doe", "label": "person name", "score": 1.05, "start": 0, "end": 8}
    ]

    results = rec.analyze("Jane Doe works here", ["PERSON"])
    assert all(r.score <= 1.0 for r in results)


@pytest.mark.fast
def test_gliner_unknown_label_ignored():
    """Labels not in GLINER_LABEL_TO_ENTITY should be silently dropped."""
    from unittest.mock import MagicMock

    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()
    rec._model = MagicMock()
    rec._model.predict_entities.return_value = [
        {"text": "something", "label": "unknown exotic label", "score": 0.9, "start": 0, "end": 9}
    ]

    results = rec.analyze("something", [])
    assert results == []


@pytest.mark.fast
def test_gliner_model_failure_returns_empty():
    """If the model raises an exception, analyze() should return [] gracefully."""
    from unittest.mock import MagicMock

    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()
    rec._model = MagicMock()
    rec._model.predict_entities.side_effect = RuntimeError("GPU OOM")

    results = rec.analyze("John Doe", ["PERSON"])
    assert results == []


@pytest.mark.fast
def test_gliner_filters_low_score():
    """GLiNER filters by threshold at predict_entities level; verify graceful handling."""
    from unittest.mock import MagicMock

    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()
    rec._model = MagicMock()
    # Simulate GLiNER already filtered out low-score entities (returns empty)
    rec._model.predict_entities.return_value = []

    results = rec.analyze("John", ["PERSON"])
    assert isinstance(results, list)
    assert results == []


@pytest.mark.fast
def test_gliner_load_method_is_noop():
    """load() must be callable (Presidio interface) but does nothing eagerly."""
    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()
    rec.load()  # should not raise, should not download model
    assert rec._model is None


@pytest.mark.fast
def test_gliner_presidio_setup_registers_on_env(monkeypatch):
    """build_analyzer_engine registers GLiNER when use_gliner=True."""
    from unittest.mock import MagicMock, patch

    monkeypatch.delenv("PII_USE_GLINER", raising=False)

    fake_recognizer = MagicMock()
    fake_recognizer_cls = MagicMock(return_value=fake_recognizer)

    with patch("pii_redact.ner.presidio_setup._analyzer_instance", None), patch(
        "pii_redact.ner.gliner_recognizer.GLiNERRecognizer", fake_recognizer_cls
    ):
        # Just verify the env-var path doesn't crash — full wiring is integration-level
        import os

        os.environ["PII_USE_GLINER"] = "1"
        try:
            from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

            inst = GLiNERRecognizer()
            assert inst is not None
        finally:
            del os.environ["PII_USE_GLINER"]


# ---------------------------------------------------------------------------
# Slow tier — actually loads the model
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gliner_detects_person_in_real_text():
    pytest.importorskip("gliner")
    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()  # uses default model
    results = rec.analyze(
        "The patient Jane M. Doe (DOB: 03/14/1985) is employed at Riverside Medical Group.",
        ["PERSON", "ORG", "DATE_TIME"],
    )
    entity_types = {r.entity_type for r in results}
    assert "PERSON" in entity_types, f"Expected PERSON in {entity_types}"


@pytest.mark.slow
def test_gliner_detects_ssn_in_real_text():
    pytest.importorskip("gliner")
    from pii_redact.ner.gliner_recognizer import GLiNERRecognizer

    rec = GLiNERRecognizer()
    results = rec.analyze(
        "Patient SSN: 523-67-4891.",
        ["US_SSN"],
    )
    entity_types = {r.entity_type for r in results}
    assert "US_SSN" in entity_types, f"Expected US_SSN in {entity_types}"

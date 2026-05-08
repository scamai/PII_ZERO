"""Tests for presidio_setup.py — singleton engines, reset, and structured engine."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Ensure engine singletons are cleared before and after every test."""
    from pii_redact.ner.presidio_setup import reset_engines
    reset_engines()
    yield
    reset_engines()


# ---------------------------------------------------------------------------
# AnalyzerEngine singleton
# ---------------------------------------------------------------------------

class TestAnalyzerEngineSingleton:
    def test_returns_same_instance_on_repeated_calls(self):
        from pii_redact.ner.presidio_setup import build_analyzer_engine
        e1 = build_analyzer_engine()
        e2 = build_analyzer_engine()
        assert e1 is e2

    def test_reset_breaks_singleton(self):
        from pii_redact.ner.presidio_setup import build_analyzer_engine, reset_engines
        e1 = build_analyzer_engine()
        reset_engines()
        e2 = build_analyzer_engine()
        assert e1 is not e2

    def test_custom_recognizers_registered(self):
        from pii_redact.ner.presidio_setup import build_analyzer_engine
        engine = build_analyzer_engine()
        names = {r.name for r in engine.registry.recognizers}
        for expected in (
            "SSN_Recognizer",
            "NPI_Recognizer",
            "Phone_Recognizer",
            "ITIN_Recognizer",
            "Passport_Recognizer",
        ):
            assert expected in names, f"{expected} not registered"

    def test_noisy_builtins_removed(self):
        from pii_redact.ner.presidio_setup import build_analyzer_engine
        engine = build_analyzer_engine()
        names = {r.name for r in engine.registry.recognizers}
        for removed in (
            "UsBankRecognizer",
            "CreditCardRecognizer",
            "PhoneRecognizer",
            "UsItinRecognizer",
            "UsPassportRecognizer",
        ):
            assert removed not in names, f"{removed} should have been removed"

    def test_analyzes_ssn(self):
        from pii_redact.ner.presidio_setup import build_analyzer_engine
        engine = build_analyzer_engine()
        results = engine.analyze("SSN 123-45-6789", language="en")
        types = {r.entity_type for r in results}
        assert "SSN" in types

    def test_analyzes_itin(self):
        from pii_redact.ner.presidio_setup import build_analyzer_engine
        engine = build_analyzer_engine()
        results = engine.analyze("ITIN 912-34-5678", language="en")
        types = {r.entity_type for r in results}
        assert "US_ITIN" in types

    def test_analyzes_passport(self):
        from pii_redact.ner.presidio_setup import build_analyzer_engine
        engine = build_analyzer_engine()
        results = engine.analyze("passport A12345678", language="en")
        types = {r.entity_type for r in results}
        assert "US_PASSPORT" in types


# ---------------------------------------------------------------------------
# StructuredEngine
# ---------------------------------------------------------------------------

class TestStructuredEngine:
    def test_returns_structured_engine_or_none(self):
        """Should return StructuredEngine if presidio-structured is installed, else None."""
        from pii_redact.ner.presidio_setup import build_structured_engine
        engine = build_structured_engine()
        if engine is None:
            pytest.skip("presidio-structured not installed")
        try:
            from presidio_structured import StructuredEngine
        except ImportError:
            pytest.skip("presidio-structured not importable")
        assert isinstance(engine, StructuredEngine)

    def test_structured_engine_singleton(self):
        from pii_redact.ner.presidio_setup import build_structured_engine
        e1 = build_structured_engine()
        e2 = build_structured_engine()
        if e1 is None:
            pytest.skip("presidio-structured not installed")
        assert e1 is e2

    def test_structured_engine_reset(self):
        from pii_redact.ner.presidio_setup import build_structured_engine, reset_engines
        e1 = build_structured_engine()
        if e1 is None:
            pytest.skip("presidio-structured not installed")
        reset_engines()
        e2 = build_structured_engine()
        assert e1 is not e2

    def test_structured_engine_uses_custom_analyzer(self):
        """StructuredEngine should carry our custom recognizer set."""
        from pii_redact.ner.presidio_setup import build_structured_engine
        engine = build_structured_engine()
        if engine is None:
            pytest.skip("presidio-structured not installed")
        names = {r.name for r in engine.analyzer_engine.registry.recognizers}
        assert "SSN_Recognizer" in names
        assert "Phone_Recognizer" in names

    def test_structured_engine_anonymizes_dataframe(self):
        """End-to-end: StructuredEngine anonymizes SSN in a pandas DataFrame column."""
        pytest.importorskip("presidio_structured")
        pytest.importorskip("pandas")
        import pandas as pd
        from presidio_anonymizer import OperatorConfig
        from presidio_structured import PandasAnalysisBuilder
        from pii_redact.ner.presidio_setup import build_structured_engine

        engine = build_structured_engine()
        if engine is None:
            pytest.skip("presidio-structured not installed")

        df = pd.DataFrame({"ssn": ["123-45-6789", "987-65-4321"], "name": ["Alice", "Bob"]})
        analysis = PandasAnalysisBuilder().generate_analysis(df)
        result_df = engine.anonymize(df, analysis, operators={"DEFAULT": OperatorConfig("replace")})
        assert result_df is not None
        # SSNs should be replaced, names left intact (no PERSON recognizer fires on "Alice")
        assert "123-45-6789" not in result_df["ssn"].values

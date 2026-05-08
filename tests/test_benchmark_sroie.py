"""Fast unit tests for the SROIE/CORD benchmark runner.

All tests are marked ``fast`` and mock heavy calls (Surya OCR, Presidio,
HuggingFace datasets) so they run in well under 5 seconds with no network
access and no GPU/model-weight requirements.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

# Ensure project root is on path regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Import benchmark modules under test (no heavy side-effects at import time)
# ---------------------------------------------------------------------------

from scripts.run_benchmark_sroie import (  # noqa: E402
    _cer,
    _extract_cord_gold_entities,
    _overlaps,
    compute_f1,
    print_table,
)


# ===========================================================================
# 1. compute_f1 — basic exact matching
# ===========================================================================


class TestComputeF1:
    def test_perfect_match_exact(self):
        gold = ["Alice", "Bob"]
        pred = ["Alice", "Bob"]
        m = compute_f1(pred, gold, partial=False)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_empty_pred_gives_zero_precision_zero_f1(self):
        m = compute_f1([], ["Alice"], partial=False)
        assert m["f1"] == 0.0
        assert m["precision"] == 0.0
        assert m["fn"] == 1

    def test_empty_gold_gives_zero_recall(self):
        m = compute_f1(["Alice"], [], partial=False)
        assert m["recall"] == 0.0
        assert m["fp"] == 1

    def test_partial_match_counts_substrings(self):
        # "Jane Doe" partially matches "Jane"
        gold = ["Jane Doe"]
        pred = ["Jane"]
        m = compute_f1(pred, gold, partial=True)
        assert m["tp"] == 1
        assert m["f1"] > 0.0

    def test_no_overlap_partial(self):
        m = compute_f1(["foo"], ["bar"], partial=True)
        assert m["f1"] == 0.0
        assert m["tp"] == 0


# ===========================================================================
# 2. _overlaps helper
# ===========================================================================


class TestOverlaps:
    def test_exact_match(self):
        assert _overlaps("hello", "hello") is True

    def test_substring_a_in_b(self):
        assert _overlaps("John", "John Smith") is True

    def test_substring_b_in_a(self):
        assert _overlaps("John Smith", "John") is True

    def test_no_overlap(self):
        assert _overlaps("Alice", "Bob") is False

    def test_empty_strings(self):
        assert _overlaps("", "Alice") is False
        assert _overlaps("Alice", "") is False


# ===========================================================================
# 3. _cer — Character Error Rate
# ===========================================================================


class TestCer:
    def test_identical_strings(self):
        assert _cer("hello", "hello") == 0.0

    def test_completely_wrong(self):
        # All chars substituted
        cer = _cer("abc", "xyz")
        assert cer == 1.0

    def test_empty_ref_returns_zero(self):
        assert _cer("", "anything") == 0.0

    def test_partial_error(self):
        # One char off: "helo" vs "hello" — 1 insertion / 5 ref chars = 0.2
        cer = _cer("hello", "helo")
        assert 0 < cer <= 0.3

    def test_case_sensitive(self):
        # _cer is case-sensitive; callers lowercase before comparison
        cer = _cer("Hello", "hello")
        assert cer > 0.0


# ===========================================================================
# 4. _extract_cord_gold_entities
# ===========================================================================


class TestExtractCordGoldEntities:
    def _make_gt(self, total_price="1,000", subtotal=None):
        parse: dict = {
            "menu": [{"nm": "Item A", "cnt": "1 x", "price": "1,000"}],
            "total": {"total_price": total_price},
        }
        if subtotal is not None:
            parse["sub_total"] = subtotal
        return {"gt_parse": parse}

    def test_extracts_total_price(self):
        gt = self._make_gt("99,000")
        entities = _extract_cord_gold_entities(gt)
        types = [t for t, _ in entities]
        values = [v for _, v in entities]
        assert "TOTAL" in types
        assert "99,000" in values

    def test_extracts_sub_total_prices(self):
        gt = self._make_gt(
            subtotal={"subtotal_price": "80,000", "tax_price": "8,000", "service_price": "12,000"}
        )
        entities = _extract_cord_gold_entities(gt)
        values = [v for _, v in entities]
        assert "80,000" in values or "8,000" in values

    def test_empty_gt_returns_empty(self):
        assert _extract_cord_gold_entities({}) == []

    def test_missing_total_key(self):
        gt = {"gt_parse": {"menu": []}}
        entities = _extract_cord_gold_entities(gt)
        assert isinstance(entities, list)

    def test_none_values_skipped(self):
        gt = {"gt_parse": {"total": {"total_price": None}}}
        entities = _extract_cord_gold_entities(gt)
        assert all(v for _, v in entities)


# ===========================================================================
# 5. run_sroie with fully mocked dependencies
# ===========================================================================


class TestRunSroieMocked:
    """Verify run_sroie logic without loading models or network."""

    def _make_fake_ds(self, n: int = 5):
        """Return a list of fake dataset rows."""
        from PIL import Image

        rows = []
        for i in range(n):
            img = Image.new("RGB", (100, 30), color=(255, 255, 255))
            rows.append({"image": img, "text": f"WORD{i}"})
        return rows

    @mock.patch("scripts.run_benchmark_sroie._get_ocr")
    @mock.patch("scripts.run_benchmark_sroie.run_ner")
    def test_run_sroie_returns_expected_keys(self, mock_ner, mock_get_ocr, tmp_path):
        from scripts.run_benchmark_sroie import run_sroie

        # Mock OCR to return a single-line result
        fake_lines = [{"text": "WORD0", "bbox": [0, 0, 50, 20], "confidence": 0.9, "source": "surya_ocr"}]
        mock_ocr_fn = mock.Mock(return_value=fake_lines)
        mock_get_ocr.return_value = mock_ocr_fn
        mock_ner.return_value = []

        fake_ds = self._make_fake_ds(3)

        # Patch the dataset loading inside run_sroie
        with mock.patch("scripts.run_benchmark_sroie.Path.exists", return_value=False), \
             mock.patch("datasets.load_dataset", return_value=fake_ds), \
             mock.patch("datasets.load_from_disk", side_effect=Exception("no cache")):
            result = run_sroie(tmp_path, max_docs=3, save_cache=False)

        assert "dataset" in result
        assert "n_docs" in result
        assert "avg_cer" in result
        assert result["n_docs"] <= 3


# ===========================================================================
# 6. run_cord with fully mocked dependencies
# ===========================================================================


class TestRunCordMocked:
    """Verify run_cord logic without loading models or network."""

    def _make_fake_ds(self, n: int = 3):
        from PIL import Image

        rows = []
        for i in range(n):
            img = Image.new("RGB", (400, 600), color=(255, 255, 255))
            gt = json.dumps({
                "gt_parse": {
                    "menu": [{"nm": f"Item {i}", "cnt": "1 x", "price": f"{i * 1000}"}],
                    "total": {"total_price": f"{i * 1000}"},
                    "sub_total": {"subtotal_price": f"{i * 900}", "tax_price": f"{i * 100}"},
                }
            })
            rows.append({"image": img, "ground_truth": gt})
        return rows

    @mock.patch("scripts.run_benchmark_sroie._get_ocr")
    @mock.patch("scripts.run_benchmark_sroie.run_ner")
    def test_run_cord_returns_expected_keys(self, mock_ner, mock_get_ocr, tmp_path):
        from scripts.run_benchmark_sroie import run_cord

        import numpy as np
        fake_lines = [
            {"text": "RECEIPT", "bbox": [0, 0, 100, 20], "confidence": 0.9, "source": "surya_ocr"},
            {"text": "1,000", "bbox": [0, 30, 60, 50], "confidence": 0.95, "source": "surya_ocr"},
        ]
        mock_get_ocr.return_value = mock.Mock(return_value=fake_lines)
        mock_ner.return_value = ["1,000"]

        fake_ds = self._make_fake_ds(3)

        with mock.patch("scripts.run_benchmark_sroie.Path.exists", return_value=False), \
             mock.patch("datasets.load_dataset", return_value=fake_ds), \
             mock.patch("datasets.load_from_disk", side_effect=Exception("no cache")):
            result = run_cord(tmp_path, max_docs=3, save_cache=False)

        assert "dataset" in result
        assert "n_docs" in result
        assert "overall" in result
        assert "per_entity" in result


# ===========================================================================
# 7. surya_ocr module — import and interface sanity (no model load)
# ===========================================================================


class TestSuryaOcrModule:
    """Verify the surya_ocr module can be imported and the public API exists."""

    def test_module_importable(self):
        import pii_redact.layers.surya_ocr as m
        assert hasattr(m, "ocr_image")
        assert hasattr(m, "_load_models")

    def test_ocr_image_bad_input_returns_empty(self):
        # Our implementation catches all errors and returns [] (graceful degradation)
        from pii_redact.layers.surya_ocr import ocr_image
        result = ocr_image("not_an_image")  # type: ignore[arg-type]
        assert result == []

    def test_ocr_image_empty_detection_returns_empty_list(self):
        """If _rec_predictor returns no text lines, ocr_image returns []."""
        import pii_redact.layers.surya_ocr as m

        fake_line_result = SimpleNamespace(text_lines=[])
        with mock.patch.object(m, "_load_models"), \
             mock.patch.object(m, "_det_predictor", mock.Mock()), \
             mock.patch.object(m, "_rec_predictor", mock.Mock(return_value=[fake_line_result])):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = m.ocr_image(img)
        assert result == []

    def test_ocr_image_returns_correct_dict_shape(self):
        """Full mock — check returned dict has text/bbox/confidence/source."""
        import pii_redact.layers.surya_ocr as m

        fake_polygon = [[10, 5], [90, 5], [90, 25], [10, 25]]
        fake_line = SimpleNamespace(
            polygon=fake_polygon,
            text="TOTAL: 42.00",
            confidence=0.97,
        )
        fake_result = SimpleNamespace(text_lines=[fake_line])

        with mock.patch.object(m, "_load_models"), \
             mock.patch.object(m, "_det_predictor", mock.Mock()), \
             mock.patch.object(m, "_rec_predictor", mock.Mock(return_value=[fake_result])):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            result = m.ocr_image(img)

        assert len(result) == 1
        r = result[0]
        assert r["text"] == "TOTAL: 42.00"
        assert "text" in r and "bbox" in r and "confidence" in r
        assert isinstance(r["bbox"], list) and len(r["bbox"]) == 4
        assert 0.9 < r["confidence"] <= 1.0

    def test_ocr_image_accepts_pil_image(self):
        """PIL Image input should be accepted and return a list."""
        import pii_redact.layers.surya_ocr as m
        from PIL import Image

        fake_result = SimpleNamespace(text_lines=[])
        with mock.patch.object(m, "_load_models"), \
             mock.patch.object(m, "_det_predictor", mock.Mock()), \
             mock.patch.object(m, "_rec_predictor", mock.Mock(return_value=[fake_result])):
            pil_img = Image.new("RGB", (100, 100), color="white")
            result = m.ocr_image(pil_img)
        assert isinstance(result, list)


# ===========================================================================
# 8. print_table smoke test
# ===========================================================================


class TestPrintTable:
    def test_prints_error_result(self, capsys):
        print_table({"dataset": "TEST", "error": "something went wrong"})
        out = capsys.readouterr().out
        assert "ERROR" in out

    def test_prints_sroie_result(self, capsys):
        result = {
            "dataset": "SROIE_word_recognition",
            "n_docs": 10,
            "elapsed_s": 5.2,
            "ocr_errors": 1,
            "avg_cer": 0.05,
            "ner_entity_counts": {"(any)": 3},
        }
        print_table(result)
        out = capsys.readouterr().out
        assert "CER" in out
        assert "10" in out

    def test_prints_cord_result(self, capsys):
        result = {
            "dataset": "CORD_receipt",
            "n_docs": 5,
            "elapsed_s": 12.3,
            "ocr_errors": 0,
            "avg_ocr_lines": 18.4,
            "match_mode": "partial",
            "overall": {"precision": 0.6, "recall": 0.5, "f1": 0.545, "tp": 5, "fp": 3, "fn": 5},
            "per_entity": {
                "TOTAL": {"precision": 0.6, "recall": 0.5, "f1": 0.545, "tp": 5, "fp": 3, "fn": 5}
            },
        }
        print_table(result)
        out = capsys.readouterr().out
        assert "CORD_receipt" in out
        assert "TOTAL" in out

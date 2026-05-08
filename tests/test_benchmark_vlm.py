"""Fast mocked tests for the VLM-augmented CORD benchmark (--vlm flag).

All tests are marked ``fast`` — no model weights, no GPU, no network.
All ML calls are patched via unittest.mock.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.fast

from pii_redact.models import RedactionBox


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_box(text_found: str, entity_type: str = "PERSON") -> RedactionBox:
    return RedactionBox(
        x=0, y=0, w=100, h=20,
        entity_type=entity_type,
        confidence=0.9,
        source="vlm",
        page=0,
        text_found=text_found,
    )


def _call_run_ner_vlm(vlm_boxes, ner_spans, pil_image=None, tmp_path=None):
    """Helper: patch _get_vlm and run_ner, then call run_ner_vlm."""
    from PIL import Image
    from scripts.run_benchmark_sroie import run_ner_vlm

    if pil_image is None:
        pil_image = Image.new("RGB", (400, 600), color="white")

    mock_extractor = mock.Mock()
    mock_extractor.extract.return_value = vlm_boxes

    with mock.patch("scripts.run_benchmark_sroie._get_vlm", return_value=mock_extractor), \
         mock.patch("scripts.run_benchmark_sroie.run_ner", return_value=ner_spans):
        return run_ner_vlm(pil_image, ocr_text="dummy text", tmp_dir=tmp_path)


# ===========================================================================
# 1. run_ner_vlm — merge and dedup logic
# ===========================================================================


class TestRunNerVlm:
    def test_vlm_only_result_included(self, tmp_path):
        boxes = [_make_box("Jane Doe")]
        result = _call_run_ner_vlm(boxes, [], tmp_path=tmp_path)
        assert "Jane Doe" in result

    def test_ner_only_result_included(self, tmp_path):
        result = _call_run_ner_vlm([], ["$42.00"], tmp_path=tmp_path)
        assert "$42.00" in result

    def test_merge_distinct_spans_from_both_sources(self, tmp_path):
        boxes = [_make_box("Alice")]
        result = _call_run_ner_vlm(boxes, ["555-1234"], tmp_path=tmp_path)
        assert "Alice" in result
        assert "555-1234" in result

    def test_dedup_identical_text(self, tmp_path):
        boxes = [_make_box("TOTAL 99.00")]
        result = _call_run_ner_vlm(boxes, ["TOTAL 99.00"], tmp_path=tmp_path)
        assert result.count("TOTAL 99.00") == 1

    def test_dedup_is_case_insensitive(self, tmp_path):
        boxes = [_make_box("TOTAL 99.00")]
        result = _call_run_ner_vlm(boxes, ["total 99.00"], tmp_path=tmp_path)
        assert len(result) == 1
        assert result[0] == "TOTAL 99.00"  # VLM (first seen) wins

    def test_vlm_empty_text_found_excluded(self, tmp_path):
        boxes = [_make_box(""), _make_box("  "), _make_box("Bob")]
        result = _call_run_ner_vlm(boxes, [], tmp_path=tmp_path)
        assert result == ["Bob"]

    def test_vlm_failure_falls_back_to_ner(self, tmp_path):
        from PIL import Image
        from scripts.run_benchmark_sroie import run_ner_vlm

        pil_img = Image.new("RGB", (400, 600))
        mock_extractor = mock.Mock()
        mock_extractor.extract.side_effect = RuntimeError("GPU OOM")

        with mock.patch("scripts.run_benchmark_sroie._get_vlm", return_value=mock_extractor), \
             mock.patch("scripts.run_benchmark_sroie.run_ner", return_value=["$9.99"]):
            result = run_ner_vlm(pil_img, ocr_text="text", tmp_dir=tmp_path)
        assert result == ["$9.99"]

    def test_tmp_png_cleaned_up(self, tmp_path):
        from PIL import Image
        from scripts.run_benchmark_sroie import run_ner_vlm

        pil_img = Image.new("RGB", (100, 100))
        mock_extractor = mock.Mock()
        mock_extractor.extract.return_value = []

        with mock.patch("scripts.run_benchmark_sroie._get_vlm", return_value=mock_extractor), \
             mock.patch("scripts.run_benchmark_sroie.run_ner", return_value=[]):
            run_ner_vlm(pil_img, ocr_text="", tmp_dir=tmp_path)

        png_files = list(tmp_path.glob("_vlm_tmp_*.png"))
        assert png_files == [], f"Temp PNG not cleaned up: {png_files}"

    def test_tmp_png_cleaned_up_on_vlm_exception(self, tmp_path):
        from PIL import Image
        from scripts.run_benchmark_sroie import run_ner_vlm

        pil_img = Image.new("RGB", (100, 100))
        mock_extractor = mock.Mock()
        mock_extractor.extract.side_effect = RuntimeError("crash")

        with mock.patch("scripts.run_benchmark_sroie._get_vlm", return_value=mock_extractor), \
             mock.patch("scripts.run_benchmark_sroie.run_ner", return_value=[]):
            run_ner_vlm(pil_img, ocr_text="", tmp_dir=tmp_path)

        png_files = list(tmp_path.glob("_vlm_tmp_*.png"))
        assert png_files == []

    def test_multiple_vlm_boxes_all_included(self, tmp_path):
        boxes = [_make_box("Alice"), _make_box("123-45-6789", "SSN"), _make_box("555-0100", "PHONE")]
        result = _call_run_ner_vlm(boxes, [], tmp_path=tmp_path)
        assert len(result) == 3

    def test_vlm_spans_appear_before_ner_spans(self, tmp_path):
        boxes = [_make_box("VLM_FIRST")]
        result = _call_run_ner_vlm(boxes, ["NER_SECOND"], tmp_path=tmp_path)
        assert result.index("VLM_FIRST") < result.index("NER_SECOND")


# ===========================================================================
# 2. run_cord — use_vlm parameter wiring
# ===========================================================================


class TestRunCordVlmMode:
    def _make_fake_ds(self, n: int = 2):
        from PIL import Image

        rows = []
        for i in range(n):
            img = Image.new("RGB", (400, 600), color="white")
            gt = json.dumps({
                "gt_parse": {"total": {"total_price": f"{i * 1000}"}}
            })
            rows.append({"image": img, "ground_truth": gt})
        return rows

    @mock.patch("scripts.run_benchmark_sroie._get_ocr")
    @mock.patch("scripts.run_benchmark_sroie.run_ner_vlm")
    def test_vlm_mode_calls_run_ner_vlm(self, mock_ner_vlm, mock_get_ocr, tmp_path):
        from scripts.run_benchmark_sroie import run_cord

        fake_lines = [{"text": "1,000", "bbox": [0, 0, 60, 20], "confidence": 0.9, "source": "surya_ocr"}]
        mock_get_ocr.return_value = mock.Mock(return_value=fake_lines)
        mock_ner_vlm.return_value = ["1,000"]

        fake_ds = self._make_fake_ds(2)
        with mock.patch("scripts.run_benchmark_sroie.Path.exists", return_value=False), \
             mock.patch("datasets.load_dataset", return_value=fake_ds), \
             mock.patch("datasets.load_from_disk", side_effect=Exception("no cache")):
            result = run_cord(tmp_path, max_docs=2, save_cache=False, use_vlm=True)

        assert mock_ner_vlm.called
        assert result.get("vlm_mode") is True

    @mock.patch("scripts.run_benchmark_sroie._get_ocr")
    @mock.patch("scripts.run_benchmark_sroie.run_ner")
    def test_default_mode_does_not_call_vlm(self, mock_ner, mock_get_ocr, tmp_path):
        from scripts.run_benchmark_sroie import run_cord

        fake_lines = [{"text": "1,000", "bbox": [0, 0, 60, 20], "confidence": 0.9, "source": "surya_ocr"}]
        mock_get_ocr.return_value = mock.Mock(return_value=fake_lines)
        mock_ner.return_value = []

        fake_ds = self._make_fake_ds(2)
        with mock.patch("scripts.run_benchmark_sroie.Path.exists", return_value=False), \
             mock.patch("datasets.load_dataset", return_value=fake_ds), \
             mock.patch("datasets.load_from_disk", side_effect=Exception("no cache")):
            with mock.patch("scripts.run_benchmark_sroie._get_vlm") as mock_get_vlm:
                result = run_cord(tmp_path, max_docs=2, save_cache=False, use_vlm=False)
                mock_get_vlm.assert_not_called()

        assert result.get("vlm_mode") is False

    @mock.patch("scripts.run_benchmark_sroie._get_ocr")
    @mock.patch("scripts.run_benchmark_sroie.run_ner")
    def test_result_dict_has_vlm_mode_key(self, mock_ner, mock_get_ocr, tmp_path):
        from scripts.run_benchmark_sroie import run_cord

        fake_lines = [{"text": "test", "bbox": [0, 0, 60, 20], "confidence": 0.9, "source": "surya_ocr"}]
        mock_get_ocr.return_value = mock.Mock(return_value=fake_lines)
        mock_ner.return_value = []

        fake_ds = self._make_fake_ds(2)
        with mock.patch("scripts.run_benchmark_sroie.Path.exists", return_value=False), \
             mock.patch("datasets.load_dataset", return_value=fake_ds), \
             mock.patch("datasets.load_from_disk", side_effect=Exception("no cache")):
            result = run_cord(tmp_path, max_docs=2, save_cache=False)

        assert "vlm_mode" in result


# ===========================================================================
# 3. print_table — VLM mode header label
# ===========================================================================


class TestPrintTableVlmMode:
    def _cord_result(self, vlm_mode: bool) -> dict:
        return {
            "dataset": "CORD_receipt",
            "n_docs": 3,
            "elapsed_s": 8.0,
            "ocr_errors": 0,
            "avg_ocr_lines": 15.0,
            "match_mode": "partial",
            "vlm_mode": vlm_mode,
            "overall": {"precision": 0.7, "recall": 0.6, "f1": 0.646,
                        "tp": 6, "fp": 2, "fn": 4},
            "per_entity": {},
        }

    def test_vlm_mode_shows_vlm_label(self, capsys):
        from scripts.run_benchmark_sroie import print_table
        print_table(self._cord_result(vlm_mode=True))
        out = capsys.readouterr().out
        assert "VLM+NER" in out

    def test_ner_only_mode_shows_ner_label(self, capsys):
        from scripts.run_benchmark_sroie import print_table
        print_table(self._cord_result(vlm_mode=False))
        out = capsys.readouterr().out
        assert "NER" in out
        assert "VLM+NER" not in out

    def test_sroie_result_has_no_mode_label(self, capsys):
        from scripts.run_benchmark_sroie import print_table
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
        assert "VLM+NER" not in out
        assert "[NER]" not in out

    def test_old_result_without_vlm_mode_key_renders_ner(self, capsys):
        """Pre-existing result JSON files without vlm_mode key default to [NER]."""
        from scripts.run_benchmark_sroie import print_table
        result = self._cord_result(vlm_mode=False)
        del result["vlm_mode"]
        print_table(result)
        out = capsys.readouterr().out
        assert "NER" in out

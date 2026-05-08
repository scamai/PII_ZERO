"""Fast unit tests for VLM grounding output parser.

Tests _parse_vlm_output — the JSON → RedactionBox converter.
No model loading; all tests run in milliseconds.
"""

from __future__ import annotations

import json
import pytest

pytestmark = pytest.mark.fast

from pii_redact.ner.vlm_extractor import _parse_vlm_output


IMG_W, IMG_H = 800, 1000  # example image size for coordinate tests


class TestParseVlmOutputHappyPath:
    def _item(self, x1=100, y1=200, x2=400, y2=250, label="PERSON", text="Jane Doe", conf=0.9):
        return {"bbox_2d": [x1, y1, x2, y2], "label": label, "text": text, "confidence": conf}

    def test_single_entity_coords_scaled(self):
        raw = json.dumps([self._item(x1=500, y1=500, x2=600, y2=550)])
        boxes = _parse_vlm_output(raw, img_w=IMG_W, img_h=IMG_H)
        assert len(boxes) == 1
        b = boxes[0]
        # 500/1000 * 800 = 400, 600/1000 * 800 = 480
        assert b.x == 400
        assert b.w == 80  # 480 - 400
        assert b.entity_type == "PERSON"
        assert b.text_found == "Jane Doe"
        assert b.source == "vlm"
        assert b.page == 0

    def test_confidence_clamped_to_1(self):
        raw = json.dumps([self._item(conf=1.5)])
        boxes = _parse_vlm_output(raw, img_w=IMG_W, img_h=IMG_H)
        assert boxes[0].confidence == 1.0

    def test_confidence_clamped_to_0(self):
        raw = json.dumps([self._item(conf=-0.3)])
        boxes = _parse_vlm_output(raw, img_w=IMG_W, img_h=IMG_H)
        assert boxes[0].confidence == 0.0

    def test_label_uppercased(self):
        raw = json.dumps([self._item(label="person")])
        boxes = _parse_vlm_output(raw, img_w=IMG_W, img_h=IMG_H)
        assert boxes[0].entity_type == "PERSON"

    def test_multiple_entities(self):
        raw = json.dumps([
            self._item(label="SSN", text="123-45-6789", x1=0, y1=0, x2=200, y2=50),
            self._item(label="PERSON", text="Alice", x1=300, y1=100, x2=500, y2=150),
        ])
        boxes = _parse_vlm_output(raw, img_w=IMG_W, img_h=IMG_H)
        assert len(boxes) == 2
        types = {b.entity_type for b in boxes}
        assert types == {"SSN", "PERSON"}

    def test_page_passed_through(self):
        raw = json.dumps([self._item()])
        boxes = _parse_vlm_output(raw, page=3, img_w=IMG_W, img_h=IMG_H)
        assert boxes[0].page == 3


class TestParseVlmOutputEdgeCases:
    def test_empty_array_returns_empty(self):
        assert _parse_vlm_output("[]") == []

    def test_none_string_returns_empty(self):
        assert _parse_vlm_output("NONE") == []

    def test_empty_string_returns_empty(self):
        assert _parse_vlm_output("") == []

    def test_markdown_fences_stripped(self):
        raw = '```json\n[{"bbox_2d": [0, 0, 500, 100], "label": "PHONE", "text": "555-1234", "confidence": 0.8}]\n```'
        boxes = _parse_vlm_output(raw, img_w=1000, img_h=1000)
        assert len(boxes) == 1
        assert boxes[0].entity_type == "PHONE"

    def test_degenerate_box_skipped(self):
        # x1 == x2 → zero width
        raw = json.dumps([{"bbox_2d": [300, 100, 300, 200], "label": "SSN", "text": "x"}])
        boxes = _parse_vlm_output(raw, img_w=1000, img_h=1000)
        assert boxes == []

    def test_reversed_coords_normalized(self):
        # x2 < x1 — model sometimes gets these backwards
        raw = json.dumps([{"bbox_2d": [600, 100, 200, 200], "label": "PERSON", "text": "Bob"}])
        boxes = _parse_vlm_output(raw, img_w=1000, img_h=1000)
        assert len(boxes) == 1
        assert boxes[0].x < boxes[0].x + boxes[0].w  # x1 < x2 after normalization

    def test_missing_bbox_skipped(self):
        raw = json.dumps([{"label": "PERSON", "text": "Carol"}])
        boxes = _parse_vlm_output(raw, img_w=1000, img_h=1000)
        assert boxes == []

    def test_wrong_bbox_length_skipped(self):
        raw = json.dumps([{"bbox_2d": [100, 200], "label": "PERSON", "text": "Dave"}])
        boxes = _parse_vlm_output(raw, img_w=1000, img_h=1000)
        assert boxes == []

    def test_non_numeric_bbox_skipped(self):
        raw = json.dumps([{"bbox_2d": ["a", "b", "c", "d"], "label": "SSN", "text": "x"}])
        boxes = _parse_vlm_output(raw, img_w=1000, img_h=1000)
        assert boxes == []

    def test_missing_confidence_defaults_to_0_75(self):
        raw = json.dumps([{"bbox_2d": [0, 0, 100, 100], "label": "EMAIL", "text": "a@b.com"}])
        boxes = _parse_vlm_output(raw, img_w=1000, img_h=1000)
        assert boxes[0].confidence == 0.75

    def test_non_json_output_returns_empty(self):
        boxes = _parse_vlm_output("this is not json at all")
        assert boxes == []

    def test_json_object_not_list_returns_empty(self):
        raw = json.dumps({"bbox_2d": [0, 0, 100, 100], "label": "PERSON"})
        boxes = _parse_vlm_output(raw, img_w=1000, img_h=1000)
        assert boxes == []

    def test_coords_clamped_to_image_bounds(self):
        # Coords > 1000 (model sometimes outputs out-of-range values)
        raw = json.dumps([{"bbox_2d": [900, 900, 1100, 1100], "label": "PLATE", "text": "ABC123"}])
        boxes = _parse_vlm_output(raw, img_w=500, img_h=500)
        assert len(boxes) == 1
        b = boxes[0]
        assert b.x + b.w <= 500
        assert b.y + b.h <= 500

    def test_json_embedded_in_prose_extracted(self):
        # Model says "Here are the results: [...]" — we should still parse it
        raw = 'I found the following PII: [{"bbox_2d": [10, 10, 200, 50], "label": "SSN", "text": "123-45-6789", "confidence": 0.95}]'
        boxes = _parse_vlm_output(raw, img_w=1000, img_h=1000)
        assert len(boxes) == 1
        assert boxes[0].entity_type == "SSN"

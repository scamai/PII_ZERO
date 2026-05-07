"""Core data contracts shared across all pipeline layers."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DocumentType(str, Enum):
    PDF_STRUCTURED = "PDF_STRUCTURED"   # ACORD, CMS-1500 — has native text layer
    SCAN_FORM = "SCAN_FORM"             # scanned paper form image
    PHOTO = "PHOTO"                     # accident scene, property photograph
    ID_DOC = "ID_DOC"                   # driver's license, passport, insurance card
    MEDICAL = "MEDICAL"                 # EOB, medical bill, clinical notes
    UNKNOWN = "UNKNOWN"                 # fallback — full-pipeline treatment


@dataclass
class RedactionBox:
    """One region to solid-fill in the output document."""
    x: int
    y: int
    w: int
    h: int
    entity_type: str        # "PERSON", "SSN", "POLICY_NUM", "FACE", "PLATE", …
    confidence: float       # 0.0–1.0
    source: str             # "regex", "presidio", "insurance_ner", "yolov8", "deface", "opencv_ink"
    page: int = 0           # 0-indexed page for multi-page PDFs
    text_found: str = ""    # original text (vault-only; never written to audit log)

    def area(self) -> int:
        return self.w * self.h

    def as_dict(self, include_text: bool = False) -> dict[str, Any]:
        d = {
            "entity_type": self.entity_type,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "page": self.page,
            "bbox": [self.x, self.y, self.w, self.h],
        }
        if include_text:
            d["text"] = self.text_found
        return d


@dataclass
class MetadataStripResult:
    stripped_fields: list[str] = field(default_factory=list)
    qr_codes_found: int = 0
    barcodes_found: int = 0
    exif_gps_found: bool = False


@dataclass
class PageResult:
    page: int
    doc_type: DocumentType
    redaction_boxes: list[RedactionBox] = field(default_factory=list)
    metadata: MetadataStripResult | None = None
    processing_ms: float = 0.0


@dataclass
class DocumentResult:
    doc_id: str
    filename: str
    doc_type: DocumentType
    pages: int
    page_results: list[PageResult] = field(default_factory=list)
    total_processing_ms: float = 0.0

    @property
    def all_boxes(self) -> list[RedactionBox]:
        return [box for pr in self.page_results for box in pr.redaction_boxes]

    @property
    def entity_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for box in self.all_boxes:
            counts[box.entity_type] = counts.get(box.entity_type, 0) + 1
        return counts

    @property
    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for box in self.all_boxes:
            counts[box.source] = counts.get(box.source, 0) + 1
        return counts

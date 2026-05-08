"""Qwen3-VL visual PII extractor — GPU-first with CPU fallback.

All torch/transformers imports are lazy (inside functions). This module is
safe to import without triggering any ML library loads.

Usage:
    from pii_redact.ner.vlm_extractor import VLMExtractor
    extractor = VLMExtractor()          # no model loaded yet
    results = extractor.extract(image)  # model loads on first call

Grounding mode: Qwen3-VL outputs normalized 0-1000 bbox coordinates alongside
entity text. Coordinates are scaled to pixel space by (coord / 1000) * dim.
This is Option B — the VLM localizes PII directly rather than relying on
post-hoc OCR alignment, which is circular when OCR already succeeded.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pii_redact.models import RedactionBox

logger = logging.getLogger(__name__)

# Grounding prompt: requests JSON with bbox_2d in Qwen3-VL's native format.
# Coordinates are normalized 0-1000 (x1, y1, x2, y2 in top-left origin).
_SYSTEM_PROMPT = """\
You are a PII detection assistant for insurance claim documents.
Your job: locate ALL personally identifiable information visible in the image
and return bounding boxes for each item.

Output a JSON array only — no explanation, no markdown fences. Each element:
  {"bbox_2d": [x1, y1, x2, y2], "label": "ENTITY_TYPE", "text": "exact text", "confidence": 0.0-1.0}

Coordinates are integers 0-1000 (normalized to image size, top-left origin).

Entity types: PERSON, SSN, DOB, ADDRESS, PHONE, EMAIL, POLICY_NUM, CLAIM_REF,
NPI, EIN, DEA_NUM, US_BANK_NUMBER, ADJUSTER_ID, ICD10_CODE, CPT_CODE,
CREDIT_CARD, FACE, PLATE

If nothing found, output: []
"""

_USER_PROMPT = "Find all PII in this insurance document image and return bounding boxes."


class VLMExtractor:
    """Qwen3-VL-based visual PII extractor with lazy model loading."""

    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    MODEL_ID_FP8 = "Qwen/Qwen3-VL-8B-Instruct-FP8"

    # FP8 inference requires SM >= 8.9 (RTX 4090, H100).
    # On older Ampere GPUs (SM 8.6, e.g. RTX 3090), we must load in bfloat16.
    FP8_MIN_SM = 8.9

    def __init__(self, model_id: str | None = None, prefer_gpu: bool = True):
        self._model_id = model_id or self.MODEL_ID_FP8
        self._prefer_gpu = prefer_gpu
        self._model = None
        self._processor = None
        self._device = None

    # ------------------------------------------------------------------
    # Model loading (lazy, GPU→CPU fallback)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load model and processor. Called once on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        logger.info("Loading VLM: %s", self._model_id)

        # Resolve model_id based on GPU compute capability.
        # FP8 inference requires SM >= 8.9 (RTX 4090, H100).
        # On SM 8.6 (RTX 3090) we fall back to the base bf16 model.
        model_id = self._model_id
        dtype: Any = "auto"

        if self._prefer_gpu and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            sm = props.major + props.minor / 10.0
            vram_gb = props.total_memory / 1e9
            logger.info("GPU: %s (SM %.1f, %.1f GB VRAM)", props.name, sm, vram_gb)

            if sm < self.FP8_MIN_SM and self.MODEL_ID_FP8 in model_id:
                model_id = self.MODEL_ID  # swap FP8 → base model
                dtype = torch.bfloat16
                logger.info(
                    "SM %.1f < %.1f: FP8 needs Ada/Hopper — using base model %s in bfloat16",
                    sm, self.FP8_MIN_SM, model_id,
                )

            try:
                # Explicit single-GPU placement beats device_map="auto": accelerate's
                # infer_auto_device_map() can conservatively resolve to CPU even when
                # the model fits (16 GB bf16 vs 25.3 GB VRAM on RTX 3090).
                self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map={"": 0},
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
                self._device = "cuda"
                used_gb = torch.cuda.memory_allocated(0) / 1e9
                logger.info(
                    "VLM loaded on GPU 0 (model=%s, dtype=%s, VRAM=%.1f GB)",
                    model_id, dtype, used_gb,
                )
            except Exception as exc:
                logger.warning("GPU load failed (%s: %s) — falling back to CPU", type(exc).__name__, exc)
                self._model = None
                self._device = None

        if self._model is None:
            # CPU: always use the base (non-FP8) model in float32
            base_id = self.MODEL_ID
            logger.info("Loading VLM on CPU in float32 (model=%s)", base_id)
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_id,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            self._device = "cpu"
            logger.info("VLM loaded on CPU (float32)")

        self._processor = AutoProcessor.from_pretrained(
            self._model_id, trust_remote_code=True
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract(self, image_path: str | Path, page: int = 0) -> list[RedactionBox]:
        """Run the VLM on an image and return detected PII as RedactionBoxes.

        Uses Qwen3-VL grounding mode: the model outputs JSON with bbox_2d
        coordinates normalized to 0-1000 scale. Coordinates are converted to
        pixel space using the image dimensions before returning.
        """
        self._load()

        import torch
        from PIL import Image

        image = Image.open(str(image_path)).convert("RGB")
        img_w, img_h = image.size

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _USER_PROMPT},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        if self._device and self._device.startswith("cuda"):
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
            )

        # Strip input tokens from output
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        raw = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        return _parse_vlm_output(raw, page=page, img_w=img_w, img_h=img_h)

    def unload(self) -> None:
        """Release GPU/CPU memory."""
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            self._processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("VLM unloaded")


# ------------------------------------------------------------------
# Output parser
# ------------------------------------------------------------------


def _parse_vlm_output(
    raw: str,
    page: int = 0,
    img_w: int = 1000,
    img_h: int = 1000,
) -> list[RedactionBox]:
    """Parse Qwen3-VL grounding JSON into RedactionBoxes with pixel coordinates.

    Expected format from model:
        [{"bbox_2d": [x1, y1, x2, y2], "label": "PERSON", "text": "Jane", "confidence": 0.9}]

    Coordinates are normalized 0-1000 (top-left origin) and scaled to pixels.
    Falls back to empty list on any parse error.
    """
    if not raw or raw.strip() in ("[]", "NONE", ""):
        return []

    # Strip markdown code fences the model sometimes adds despite instructions
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

    try:
        items = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-resort: find the first JSON array in the output
        m = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not m:
            logger.warning("VLM output not parseable as JSON: %s", raw[:200])
            return []
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            logger.warning("VLM JSON extraction failed: %s", raw[:200])
            return []

    if not isinstance(items, list):
        return []

    boxes: list[RedactionBox] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        bbox = item.get("bbox_2d")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue

        try:
            x1_n, y1_n, x2_n, y2_n = (float(c) for c in bbox)
        except (TypeError, ValueError):
            continue

        # Scale normalized 0-1000 coords to pixel space
        x1 = int(x1_n / 1000 * img_w)
        y1 = int(y1_n / 1000 * img_h)
        x2 = int(x2_n / 1000 * img_w)
        y2 = int(y2_n / 1000 * img_h)

        # Clamp and ensure x1 < x2, y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        if x2 <= x1 or y2 <= y1:
            continue  # degenerate box

        entity_type = str(item.get("label", "UNKNOWN")).upper()
        text_found = str(item.get("text", ""))
        try:
            confidence = float(item.get("confidence", 0.75))
        except (TypeError, ValueError):
            confidence = 0.75
        confidence = max(0.0, min(1.0, confidence))

        boxes.append(
            RedactionBox(
                x=x1,
                y=y1,
                w=x2 - x1,
                h=y2 - y1,
                entity_type=entity_type,
                confidence=confidence,
                source="vlm",
                page=page,
                text_found=text_found,
            )
        )

    return boxes

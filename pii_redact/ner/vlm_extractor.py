"""Qwen3-VL visual PII extractor — GPU-first with CPU fallback.

All torch/transformers imports are lazy (inside functions). This module is
safe to import without triggering any ML library loads.

Usage:
    from pii_redact.ner.vlm_extractor import VLMExtractor
    extractor = VLMExtractor()          # no model loaded yet
    results = extractor.extract(image)  # model loads on first call
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pii_redact.models import RedactionBox

logger = logging.getLogger(__name__)

# Prompt that instructs the VLM to locate PII in an insurance document image.
_SYSTEM_PROMPT = """\
You are a PII detection assistant for insurance claim documents.
Your job: identify ALL personally identifiable information visible in the image.

For each PII item found, output one line in EXACTLY this format:
  ENTITY_TYPE | text_found | confidence

Entity types to detect:
  PERSON, SSN, DOB, ADDRESS, PHONE, EMAIL, POLICY_NUM, CLAIM_REF, NPI,
  EIN, DEA_NUM, ROUTING_NUM, ADJUSTER_ID, ICD10_CODE, CPT_CODE, FACE, PLATE

Rules:
- confidence is a float 0.0–1.0
- If nothing found, output: NONE
- Do not add explanations, headers, or any other text.
"""

_USER_PROMPT = "Find all PII in this insurance document image."


class VLMExtractor:
    """Qwen3-VL-based visual PII extractor with lazy model loading."""

    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
    MODEL_ID_FP8 = "Qwen/Qwen3-VL-8B-Instruct-FP8"

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
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading VLM: %s", self._model_id)

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": "auto",
        }

        if self._prefer_gpu and torch.cuda.is_available():
            try:
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self._model_id,
                    device_map="cuda:0",
                    **load_kwargs,
                )
                self._device = "cuda:0"
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info("VLM loaded on GPU (%.1f GB VRAM)", vram_gb)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                logger.warning(
                    "GPU load failed (%s) — falling back to CPU", exc
                )
                self._model = None
                self._device = None

        if self._model is None:
            # CPU fallback — 4-bit quantization to stay within RAM budget
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_4bit=True)
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self._model_id,
                    device_map="cpu",
                    quantization_config=bnb_config,
                    **load_kwargs,
                )
                self._device = "cpu"
                logger.info("VLM loaded on CPU (4-bit quantized)")
            except Exception:
                # bitsandbytes may not support CPU — try plain CPU
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self._model_id,
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

        Note: The VLM returns entity types and text but not pixel coordinates —
        coordinates come from the downstream text-layer or OCR alignment step.
        This method returns boxes with (x=0, y=0, w=0, h=0) as placeholders;
        callers must align these against the actual text positions.
        """
        self._load()

        import torch
        from PIL import Image

        image = Image.open(str(image_path)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": _USER_PROMPT},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            system_message=_SYSTEM_PROMPT,
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
                max_new_tokens=512,
                do_sample=False,
            )

        # Strip input tokens from output
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        raw = self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()

        return _parse_vlm_output(raw, page=page)

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


def _parse_vlm_output(raw: str, page: int = 0) -> list[RedactionBox]:
    """Parse the VLM's line-delimited entity output into RedactionBoxes."""
    boxes: list[RedactionBox] = []

    for line in raw.splitlines():
        line = line.strip()
        if not line or line == "NONE":
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue

        entity_type = parts[0].upper()
        text_found = parts[1] if len(parts) > 1 else ""
        try:
            confidence = float(parts[2]) if len(parts) > 2 else 0.75
        except ValueError:
            confidence = 0.75

        confidence = max(0.0, min(1.0, confidence))

        # Coordinates are unknown at this stage — callers align them later
        boxes.append(
            RedactionBox(
                x=0, y=0, w=0, h=0,
                entity_type=entity_type,
                confidence=confidence,
                source="vlm",
                page=page,
                text_found=text_found,
            )
        )

    return boxes

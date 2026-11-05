"""Per-model inference adapters.

Every adapter exposes the same contract:

    adapter = build_adapter("qwen3vl")
    adapter.load()
    text = adapter.generate(frame_paths, prompt)   # -> raw model string

Design notes worth knowing before you edit this file:

* All three models are fed the SAME 32 JPEG frames as a multi-image input.
  None of them uses its native video ingestion. That is deliberate: it holds
  the visual input constant across three different video pipelines, and it
  matches how the VSI-Bench paper evaluated its 15 baselines.
* Each model has a small fallback ladder for the image content key, because
  `apply_chat_template` accepts different keys across transformers builds
  ("image" as a path, "url", or a PIL object). The first form that works is
  logged into every result row, so the report can state it exactly.
* Decoding is greedy (do_sample=False) for reproducibility, even though
  Gemma 4's card recommends temperature 1.0 / top_p 0.95 / top_k 64.
  Deviating from the card is the correct call for a benchmark comparison
  and is disclosed in PROTOCOL.md.
"""

from __future__ import annotations

import os
import re
from typing import Callable

MODEL_IDS = {
    "qwen3vl": "Qwen/Qwen3-VL-8B-Instruct",
    "gemma4": "google/gemma-4-12B-it",
    "molmo2": "allenai/Molmo2-8B",
}

MAX_NEW_TOKENS = int(os.environ.get("VSI_MAX_NEW_TOKENS", "48"))
ATTN = os.environ.get("VSI_ATTN", "").strip() or None


def _load_kwargs() -> dict:
    kw = {"device_map": "auto"}
    if ATTN:
        kw["attn_implementation"] = ATTN
    return kw


def _from_pretrained(cls, model_id: str, **extra):
    """Handle the dtype=/torch_dtype= rename across transformers versions."""
    base = _load_kwargs()
    base.update(extra)
    try:
        return cls.from_pretrained(model_id, dtype="auto", **base)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype="auto", **base)


_CTRL = re.compile(r"<\|?(?:turn|channel|think|start_of_turn|end_of_turn|eos|bos)\|?>",
                   re.IGNORECASE)


def strip_channels(text: str) -> str:
    """Remove reasoning-channel and turn markers left by the tokenizer.

    Gemma 4 emits an empty <|channel>thought<channel|> block when thinking is
    disabled and closes answers with <turn|>. Any of those surviving into the
    scored string breaks the official first-token parser.
    """
    if "<channel|>" in text:
        tail = text.rsplit("<channel|>", 1)[-1]
        # Only keep the tail when the answer actually follows the marker; a
        # trailing marker with nothing after it must not erase the answer.
        if tail.strip():
            text = tail
    return _CTRL.sub("", text).strip()


class BaseAdapter:
    name = "base"
    content_key_used = None

    def load(self):  # pragma: no cover - requires GPU
        raise NotImplementedError

    def generate(self, frame_paths: list[str], prompt: str) -> str:
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------
    def _ladder(self) -> list[tuple[str, Callable[[str], dict]]]:
        from PIL import Image
        return [
            ("image=path", lambda p: {"type": "image", "image": p}),
            ("url=path", lambda p: {"type": "image", "url": p}),
            ("image=PIL", lambda p: {"type": "image", "image": Image.open(p).convert("RGB")}),
        ]


class Qwen3VLAdapter(BaseAdapter):
    name = "qwen3vl"

    def load(self):
        import torch  # noqa: F401
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_id = MODEL_IDS[self.name]
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = _from_pretrained(Qwen3VLForConditionalGeneration, self.model_id)
        self.model.eval()
        return self

    def generate(self, frame_paths, prompt):
        import torch
        from qwen_vl_utils import process_vision_info

        last_err = None
        for key, make in self._ladder():
            messages = [{"role": "user", "content": [make(p) for p in frame_paths] + [{"type": "text", "text": prompt}]}]
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(text=[text], images=image_inputs,
                                        videos=video_inputs, padding=True,
                                        return_tensors="pt").to(self.model.device)
                self.content_key_used = key
                break
            except Exception as e:                      # noqa: BLE001
                last_err = e
                continue
        else:
            raise RuntimeError(f"qwen3vl: all content-key forms failed: {last_err}")

        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()


class Gemma4Adapter(BaseAdapter):
    name = "gemma4"

    def load(self):
        import torch  # noqa: F401
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.model_id = MODEL_IDS[self.name]
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = _from_pretrained(AutoModelForMultimodalLM, self.model_id)
        self.model.eval()
        return self

    def generate(self, frame_paths, prompt):
        import torch

        last_err = None
        for key, make in self._ladder():
            messages = [{"role": "user", "content": [make(p) for p in frame_paths] + [{"type": "text", "text": prompt}]}]
            try:
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, return_dict=True,
                    return_tensors="pt", add_generation_prompt=True).to(self.model.device)
                self.content_key_used = key
                break
            except Exception as e:                      # noqa: BLE001
                last_err = e
                continue
        else:
            raise RuntimeError(f"gemma4: all content-key forms failed: {last_err}")

        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False)
        # Decode with special tokens removed: this build emits turn/channel
        # markers ("B<turn|>") that the benchmark's first-token parser reads as
        # part of the answer, scoring a correct letter as wrong. lmms-eval
        # decodes with skip_special_tokens=True, so do the same. The raw form is
        # only a fallback for a build that yields nothing.
        text = self.processor.decode(out[0][input_len:], skip_special_tokens=True)
        if not text.strip():
            text = self.processor.decode(out[0][input_len:], skip_special_tokens=False)
        return strip_channels(text).strip()


class Molmo2Adapter(BaseAdapter):
    name = "molmo2"

    def load(self):
        import torch  # noqa: F401
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = MODEL_IDS[self.name]
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True)
        self.model = _from_pretrained(AutoModelForImageTextToText, self.model_id,
                                      trust_remote_code=True)
        self.model.eval()
        return self

    def generate(self, frame_paths, prompt):
        import torch

        last_err = None
        for key, make in self._ladder():
            messages = [{"role": "user", "content": [make(p) for p in frame_paths] + [{"type": "text", "text": prompt}]}]
            try:
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", return_dict=True)
                inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                          for k, v in inputs.items()}
                self.content_key_used = key
                break
            except Exception as e:                      # noqa: BLE001
                last_err = e
                continue
        else:
            raise RuntimeError(f"molmo2: all content-key forms failed: {last_err}")

        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False)
        toks = out[0, inputs["input_ids"].size(1):]
        return self.processor.tokenizer.decode(toks, skip_special_tokens=True).strip()


class StubAdapter(BaseAdapter):
    """No-GPU dry run: exercises every code path except the model itself."""

    name = "stub"

    def load(self):
        self.content_key_used = "stub"
        return self

    def generate(self, frame_paths, prompt):
        # 32 frames normally, 0 for the blind control arm.
        assert len(frame_paths) in (0, 32), f"expected 0 or 32 frames, got {len(frame_paths)}"
        assert "Options:" in prompt or "single word or phrase" in prompt
        rng = int(frame_paths[0].split("/")[-2].__hash__()) % 10 if frame_paths else 7
        if "Options:" in prompt:
            return f"{chr(65 + (rng % 3))}"
        return f"{2.0 + (rng % 7) * 0.4:.1f}"


def build_adapter(name: str) -> BaseAdapter:
    return {"qwen3vl": Qwen3VLAdapter, "gemma4": Gemma4Adapter,
            "molmo2": Molmo2Adapter, "stub": StubAdapter}[name]()

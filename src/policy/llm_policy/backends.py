"""Pluggable VLM backends for the LLM Policy baseline.

The policy is backend-agnostic: it builds a (system, user, images) request and
asks a `VLMBackend` for raw text, then parses an action out of it. Two backends:

  - `QwenLocalBackend`  — local `Qwen3-VL-2B` (the free placeholder for dev).
  - `OpenAIBackend`     — OpenAI-compatible chat API (LETSUR etc.), reusing the
                          project convention in `src/vlm/api.py`. This is what we
                          swap in for the final runs.

Both expose the same `generate(images, system, user) -> str`, so switching is a
one-line change in the eval script / config (`backend: qwen_local | api`).
"""

from __future__ import annotations

import base64
import io
import os
from abc import ABC, abstractmethod
from typing import Sequence


class VLMBackend(ABC):
    """A vision-language chat model that returns raw text for one request."""

    @abstractmethod
    def generate(self, images: Sequence, system: str, user: str) -> str:
        """Return the model's text response to (system, user, images).

        `images` are PIL.Image objects (the current frame, possibly more).
        """


def _pil_to_data_url(image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


class QwenLocalBackend(VLMBackend):
    """Local Qwen3-VL-2B (the placeholder backend — free, no API).

    Mirrors how `src/policy/vla` loads Qwen3-VL, but uses the *generation* head
    (`Qwen3VLForConditionalGeneration`) since here we want text out, not hidden
    states. Lazily loads weights on construction.
    """

    def __init__(
        self,
        repo_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        *,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        td = getattr(torch, dtype)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(repo_id, dtype=td).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(repo_id)

    def generate(self, images: Sequence, system: str, user: str) -> str:
        import torch

        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": user}]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=list(images), return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0, temperature=max(self.temperature, 1e-5),
            )
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


class OpenAIBackend(VLMBackend):
    """OpenAI-compatible chat backend (LETSUR / any OpenAI-style endpoint).

    `vlm_config` keys (same convention as `src/vlm/api.py`): api_base, model,
    api_key_env, max_tokens, temperature, retry_attempts, retry_delay_seconds.
    Default to a CHEAP model in the config — never silently use a preview/premium
    model (see project cost note).
    """

    def __init__(self, vlm_config: dict) -> None:
        self.cfg = vlm_config

    def generate(self, images: Sequence, system: str, user: str) -> str:
        import time

        from openai import OpenAI

        api_key = os.environ.get(self.cfg.get("api_key_env", "LETSUR_API_KEY"))
        if not api_key:
            raise RuntimeError(
                f"API key env '{self.cfg.get('api_key_env', 'LETSUR_API_KEY')}' is not set. "
                "Export it before running the API backend."
            )
        client = OpenAI(base_url=self.cfg["api_base"], api_key=api_key)
        content = [{"type": "text", "text": user}]
        content += [{"type": "image_url", "image_url": {"url": _pil_to_data_url(im)}} for im in images]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]

        retries = int(self.cfg.get("retry_attempts", 3))
        delay = float(self.cfg.get("retry_delay_seconds", 2))
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = client.chat.completions.create(
                    model=self.cfg["model"], messages=messages,
                    max_tokens=int(self.cfg.get("max_tokens", 512)),
                    temperature=float(self.cfg.get("temperature", 0.2)),
                )
                text = (resp.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("VLM returned empty content")
                return text
            except Exception as e:  # noqa: BLE001 — surface after retries
                last_err = e
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
        raise RuntimeError(f"OpenAI backend failed after {retries} attempts: {last_err}")


def build_backend(cfg: dict) -> VLMBackend:
    """Construct a backend from a config dict: {backend: qwen_local|api, ...}."""
    kind = cfg.get("backend", "qwen_local")
    if kind == "qwen_local":
        q = cfg.get("qwen", {})
        return QwenLocalBackend(
            q.get("repo_id", "Qwen/Qwen3-VL-2B-Instruct"),
            dtype=q.get("dtype", "bfloat16"), device=q.get("device", "cuda"),
            max_new_tokens=int(q.get("max_new_tokens", 256)), temperature=float(q.get("temperature", 0.2)),
        )
    if kind == "api":
        return OpenAIBackend(cfg["api"])
    raise ValueError(f"unknown backend: {kind!r} (expected 'qwen_local' or 'api')")


__all__ = ["VLMBackend", "QwenLocalBackend", "OpenAIBackend", "build_backend"]

"""LLM Policy baseline (issue #22) — VLM-as-policy, Photo Agent style.

Training-free: a vision-language model is prompted with the current frame + a
natural-language framing brief and returns the next camera move. Contrasts with
our method as "implicit previsualization" (the LLM imagines the move's effect)
vs. our explicit video world model. Backend is pluggable (local Qwen3-VL-2B
placeholder, OpenAI-compatible API for final runs). See REFERENCES.md.
"""

from src.policy.llm_policy.backends import OpenAIBackend, QwenLocalBackend, VLMBackend, build_backend
from src.policy.llm_policy.policy import LLMPhotoPolicy
from src.policy.llm_policy.prompt import SYSTEM_PROMPT, build_user_prompt, describe_goal

__all__ = [
    "LLMPhotoPolicy", "VLMBackend", "QwenLocalBackend", "OpenAIBackend", "build_backend",
    "SYSTEM_PROMPT", "describe_goal", "build_user_prompt",
]

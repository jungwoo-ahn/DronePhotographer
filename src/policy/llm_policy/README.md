# `src/policy/llm_policy/`

LLM Policy baseline (issue #22) — VLM-as-policy, Photo Agent style.

Same family layout as `vla/` and `diffusion_policy/`, **minus the trainer/dataset**:
this baseline is **training-free**, so there is no `trainer.py` / `dataset.py`.

- `backends.py` — pluggable `VLMBackend`: `QwenLocalBackend` (local Qwen3-VL-2B
  placeholder) and `OpenAIBackend` (OpenAI-compatible API, e.g. LETSUR, for final
  runs). `build_backend(cfg)` selects via `backend: qwen_local | api`.
- `prompt.py` — system prompt + `describe_goal` (target shot profile → natural
  language) + `build_user_prompt`.
- `policy.py` — `LLMPhotoPolicy.act(image, target) -> (action_5d, info)`.

Run it with `scripts/eval_llm_policy.py --config configs/policy/llm_policy_qwen.yaml`
— same start frame / target yaml / pose-proxy metric as the other baselines, so
results are directly comparable. Reuses `common/` (action repr, goal space,
reward) and `src/vlm/api.py` (the API convention).

# LLM Policy baseline — references & attribution

LLM-as-policy baseline (issue #22). Training-free reimplementation against our
infra, informed by:

| Source | What we took |
|---|---|
| **Photo Agent** (no public code) | The core idea: prompt a (V)LM with the current view and a framing objective, and have it propose the next camera adjustment — reasoning about the move's effect before committing. We reimplement this as a single-step `image + framing brief -> JSON camera move`. |
| **LAMP** (cyberiada.github.io/LAMP/) | Reference for LLM-driven photographic guidance / language-conditioned framing. |
| **`src/vlm/api.py`** (this repo) | The OpenAI-compatible API convention (LETSUR endpoint, `parse_vlm_json`, retry logic) reused by `OpenAIBackend`. |
| **Qwen3-VL** (`Qwen/Qwen3-VL-2B-Instruct`, Apache-2.0) | The local placeholder backend for development (free, no API). |

## Where it sits on the previsualization axis
This is the **"implicit previsualization"** baseline: the LLM is asked to imagine
how a move changes the framing, but has no explicit world model and no training.
A win for our video world-action policy over this baseline supports the claim
that *explicit* previsualization beats implicit LLM reasoning.

## What it shares with the other baselines (for comparability)
Same target shot-profile spec (the `target:` score-key yaml), same camera-local
5D action `(dx, dy, dz, dyaw, dpitch)` (`common/action_repr`), same eval entry
point and pose-proxy distance metric (`common/reward.score_distance`) as
`eval_vla_policy.py` / `eval_diffusion_policy.py`. Differences by nature: it is
**training-free** (no dataset/trainer in this family) and the goal enters as
**text** (the LLM's native interface) rather than as a vector.

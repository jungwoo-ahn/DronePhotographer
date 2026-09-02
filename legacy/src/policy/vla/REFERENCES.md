# VLA baseline — references & attribution

π0-style VLA ablation baseline (issue #22). Not a vendored codebase — a from-
scratch reimplementation against our infra, informed by:

| Source | What we took |
|---|---|
| **π0** (Physical Intelligence, "π₀: A Vision-Language-Action Flow Model for General Robot Control", 2024) — pi.website/blog/pi0, openpi (github.com/Physical-Intelligence/openpi, Apache-2.0) | The core idea: a VLM + a **flow-matching action expert** that denoises a continuous action chunk (~10 integration steps). We predict flow velocity `v = ε − a0` over a `chunk_size`-step action chunk. |
| **Qwen3-VL** (`Qwen/Qwen3-VL-2B-Instruct`, Apache-2.0) | The VLM backbone (`Qwen3VLModel`); its last-layer hidden states are the action expert's cross-attention context. |

## Faithful simplification vs. π0
π0 fuses the action expert into the VLM via **joint attention** (action-token
weights inside the same transformer). We instead run the VLM to get hidden
states for (image + minimal prompt), then a **separate ActionExpert
cross-attends** to those states (plus appended goal tokens) while self-attending
over the noisy action chunk. This is the standard π0 reimplementation pattern and
avoids surgery inside Qwen's attention. Noted so the difference is explicit.

## What makes it an *ablation* of our method
Identical to the Cosmos world-action policy in every respect except the world
model: same v7 data + windows + HER goal sampling (`common/dataset_base`), same
goal space + normalization (`common/goal_space`), same 5D action + `ACTION_SCALE`
(`common/action_repr`), same flow convention (`common/flow`), same eval metric
(`common/reward.pose_distance`). Removed: future-frame prediction (the world
model) and the value head. A win for Cosmos over this baseline is attributable to
previsualization.

# legacy/

Archive of the retired **Qwen VLM forward model + MPC inference** pipeline. The project has pivoted to a goal-conditioned Cosmos video world-action policy; see `CLAUDE.md` and `Cleanup.md` at the repo root for context.

## What's here

Paths inside `legacy/` mirror their original locations in the repo.

- `legacy/src/vlm_qwen25/` — Qwen score-prediction modules: `dataset.py`, `collator.py`, `schema.py`, `mpc.py`, `objective.py`. (`rotation_utils.py` was kept and moved to `src/utils/`; `prompt.py` was left in place pending discussion.)
- `legacy/scripts/` — Qwen training entry points (`train.py`, `train_qwen25_vl_2_h200.sh`, `train_qwen35_vl_*.sh`), Qwen eval/predict (`eval_qwen25_vl.py`, `predict_qwen25_vl.py`), MPC inference (`infer_mpc.py`, `infer_mpc_blender.py`, `run_*_mpc.sh`), and inference-side utilities (`benchmark_inference_speed.py`, `check_rotation_consistency.py`).
- `legacy/configs/` — Qwen training configs (`qwen25_vl_7b_*.yaml`, `vla_qwen3_2b*.yaml`).
- `legacy/tests/` — Tests for the modules above.

### Second wave: the hand-rolled Qwen3-VL VLA baseline (archived 2026-09-02)

Separate from the Qwen+MPC pipeline above. This was **our own** π0-style baseline —
Qwen3-VL-2B with a frozen vision tower, goal projected to soft tokens, and a flow-matching
action expert. It is retired because the baseline it stood in for is now served by two REAL
pretrained VLAs fine-tuned through LeRobot (pi0.5 and GR00T N1.7), which is a stronger and
more defensible comparison than a reimplementation. See `docs/baselines.md`.

- `legacy/src/policy/vla/` — `model.py`, `action_expert.py`, `dataset.py`, `trainer.py`.
- `legacy/scripts/` — `train_vla_policy.py`, `eval_vla_policy.py`, `check_reconstruction_vla.py`.
- `legacy/configs/` — `vla_qwen3_2b.yaml`, `vla_qwen3_2b_smoke.yaml`, `vla_qwen3_2b_dit.yaml`.
- `legacy/tests/policy/` — `test_vla_model_mock.py`, `test_vla_action_expert.py`, `test_vla_dataset.py`.

Replaced by `scripts/train_pi05.sh`, `scripts/train_groot.sh`, `scripts/export_lerobot.py`,
and `scripts/check_reconstruction_lerobot.py`.

**Not archived, still live, despite the similar names:** `src/vlm/` (the VLM-in-the-loop
object *placement* pipeline behind `scripts/vlm_place_orchestrator.py`, which produced the v6
placements the rollouts still load) and `src/policy/llm_policy/` (the training-free
VLM-as-policy baseline, which has a local-Qwen backend but is not part of this retirement).

## Why archive instead of delete

The Cosmos pipeline isn't working end-to-end yet. Keeping the previous pipeline around for reference is cheap; losing the only working implementation of e.g. the MPC objective or the score schema would be expensive. **Default to archival, not deletion.**

## Status

- Imports inside `legacy/` still reference each other (e.g. `from src.vlm_qwen25.mpc import ...`) and **will not run** as-is — they'd need import paths fixed before any module here is revived.
- No code outside `legacy/` imports from anything in this directory. If a grep contradicts that, the cleanup is incomplete.

## When to delete

Delete this directory (and the root `Cleanup.md`) once:
1. The Cosmos goal-conditioned policy trains end-to-end on a smoke-test scene, and
2. Nothing here has been consulted for ≥1 month.

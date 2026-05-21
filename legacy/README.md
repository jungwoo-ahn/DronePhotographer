# legacy/

Archive of the retired **Qwen VLM forward model + MPC inference** pipeline. The project has pivoted to a goal-conditioned Cosmos video world-action policy; see `CLAUDE.md` and `Cleanup.md` at the repo root for context.

## What's here

Paths inside `legacy/` mirror their original locations in the repo.

- `legacy/src/vlm_qwen25/` — Qwen score-prediction modules: `dataset.py`, `collator.py`, `schema.py`, `mpc.py`, `objective.py`. (`rotation_utils.py` was kept and moved to `src/utils/`; `prompt.py` was left in place pending discussion.)
- `legacy/scripts/` — Qwen training entry points (`train.py`, `train_qwen25_vl_2_h200.sh`, `train_qwen35_vl_*.sh`), Qwen eval/predict (`eval_qwen25_vl.py`, `predict_qwen25_vl.py`), MPC inference (`infer_mpc.py`, `infer_mpc_blender.py`, `run_*_mpc.sh`), and inference-side utilities (`benchmark_inference_speed.py`, `check_rotation_consistency.py`).
- `legacy/configs/` — Qwen training configs (`qwen25_vl_7b_*.yaml`).
- `legacy/tests/` — Tests for the modules above.

## Why archive instead of delete

The Cosmos pipeline isn't working end-to-end yet. Keeping the previous pipeline around for reference is cheap; losing the only working implementation of e.g. the MPC objective or the score schema would be expensive. **Default to archival, not deletion.**

## Status

- Imports inside `legacy/` still reference each other (e.g. `from src.vlm_qwen25.mpc import ...`) and **will not run** as-is — they'd need import paths fixed before any module here is revived.
- No code outside `legacy/` imports from anything in this directory. If a grep contradicts that, the cleanup is incomplete.

## When to delete

Delete this directory (and the root `Cleanup.md`) once:
1. The Cosmos goal-conditioned policy trains end-to-end on a smoke-test scene, and
2. Nothing here has been consulted for ≥1 month.

# CLEANUP.md

Migration plan for retiring the **Qwen VLM forward model + MPC inference** pipeline as the project transitions to a **goal-conditioned Cosmos video world-action policy**. See `CLAUDE.md` for project context and the rationale behind the pivot.

This document is for Claude Code (claude.ai/code) and any contributor doing the cleanup. It is transitional — once the migration is complete, this file should be deleted.

## Ground Rules

- **Always confirm with the user before deleting or relocating files.** Default to caution.
- **Prefer archival over deletion.** Move legacy files into a `legacy/` directory at the repo root rather than removing them outright. The Cosmos pipeline isn't fully working yet; the old code may need to be referenced.
- **Update imports after relocation.** Most legacy files only import each other, so this should be self-contained, but verify.
- **Don't touch preserved files.** The Blender rendering, detection, and shot-profile-scoring code is still in use.

## Definite Removal / Relocate to `legacy/`

These are part of the retired Qwen+MPC pipeline. They should not be extended, built on, or referenced by new code.

- `src/vlm_qwen25/dataset.py` — Qwen-specific score-pair dataset
- `src/vlm_qwen25/collator.py` — Qwen chat collator with prompt token masking
- `src/vlm_qwen25/schema.py` — JSON score serialization; no longer the model's output format
- `src/vlm_qwen25/mpc.py` — MPC inference; removed in new direction
- `src/vlm_qwen25/objective.py` — MPC objective; removed
- `scripts/train.py` — Qwen training entry point
- `scripts/eval_qwen25_vl.py` — Qwen eval
- `scripts/predict_qwen25_vl.py` — single-prediction script
- `scripts/infer_mpc_blender.py` — MPC inference loop
- `scripts/train_qwen25_vl_2_h200.sh` — Qwen multi-GPU launcher
- `configs/qwen25_vl_*.yaml` — Qwen training configs

## Relocate (still useful)

- `src/vlm_qwen25/rotation_utils.py` → `src/utils/rotation_utils.py`
  Contains camera frame conversions (world ↔ camera-local), Gram-Schmidt orthonormalization, and Blender camera conventions. These are still needed for the action representation in the new policy.

## Preserve As-Is

- `render_object.py` and the rest of the Blender rendering pipeline
- `src/drones/` and `src/scenes/`
- `src/detectors/` and `scripts/annotate_detections.py`
- `src/scoring/` and `scripts/score_annotations.py`
- All `outputs/` artifacts already generated

## Discuss Before Changing

- `src/vlm_qwen25/prompt.py` — currently builds action-text strings for the Qwen prompt. May be reused if an LLM-based Aesthetic Commentary module is built that maps natural language → shot profile goals. Confirm intent with the user before touching.
- Annotation JSON schema — shot profile fields stay, but the (state, action, score) pairing logic at training time changes. The on-disk annotation format may not need changes; the dataset loader is what needs replacement.
- `requirements.txt` — Qwen-specific dependencies can be removed once cleanup lands; Cosmos dependencies will be added.

## Suggested Order

1. Verify with the user that the pivot is committed and legacy code can be safely retired.
2. Create `legacy/` at the repo root.
3. Move the "definite removal" files into `legacy/`, preserving directory structure (e.g., `legacy/src/vlm_qwen25/`, `legacy/scripts/`).
4. Relocate `rotation_utils.py` from `src/vlm_qwen25/` to `src/utils/`. Update any preserved code that imports it.
5. Add stubs for `src/policy/` (dataset, model, trainer) so the new pipeline has a clear home.
6. Add `legacy/README.md` explaining what's there, why, and that it's slated for eventual deletion.
7. Once the Cosmos pipeline is working end-to-end and nothing in `legacy/` is needed for reference, delete the directory and this file.

## Verification Checklist (per cleanup pass)

- [ ] No file outside `legacy/` imports from `src/vlm_qwen25/` (except possibly `rotation_utils.py` during transition)
- [ ] No script under `scripts/` references the moved legacy entry points
- [ ] All preserved scripts still run end-to-end on a smoke-test scene
- [ ] `requirements.txt` is consistent with what's actually imported by non-legacy code
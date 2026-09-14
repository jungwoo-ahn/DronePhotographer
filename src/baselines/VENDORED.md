# Vendored baseline code

Copied from `jungwoo-ahn/DronePhotographer` **origin/main @ 30cdb168** (PR #30
`008-baseline-retrain`, merged 2026-09-02), the code the pi0.5 / GR00T / DP checkpoints in
`baseline_ckpts/` were trained against. It is here so the baselines can run through OUR
rollout loop (`scripts/closed_loop_eval.py --policy {pi05,groot,dp}`) instead of a port of it.

| here | source | change |
|---|---|---|
| `facing.py` | `src/policy/common/facing.py` | none |
| `goal_space.py` | `src/policy/common/goal_space.py` | none (`src.scoring.V5_SCORE_KEYS` is identical on both sides -- checked) |
| `goal_text.py` | `src/policy/common/goal_text.py` | two import paths |
| `dp_denoiser.py` | `src/policy/diffusion_policy/denoiser.py` | none |
| `dp_model.py` | `src/policy/diffusion_policy/model.py` + the two inference helpers from `dataset.py` | import paths; `POSE_DIM/ACTION_DIM/ACTION_SCALE` inlined from their `action_repr.py` |

**Do not "fix" these to match V12's own modules.** Where they differ from ours, the difference
is what the baselines were trained on:

* their `goal_space.DEFAULT_GOAL_KEYS` carries `cam_to_obj_azimuth_deg` (world azimuth). Neither
  trained baseline uses it -- the VLAs use `goal_text.NL_GOAL_KEYS` and DP its config's
  `goal_score_keys`, both with `subject_bearing_deg`, both in exactly our `DEFAULT_GOAL_KEYS`
  order. The adapters assert this at load time.
* their `action_repr.ACTION_DIM` is 10; ours is 9. Different constant, different purpose.

What is NOT vendored, because it is identical: `reward.py` (distance functions byte-equal;
ours only appends `VALUE_SCALE`), `apply_action_9d` / `encode_action_9d` (function bodies
AST-equal), `V5_SCORE_KEYS`.

The `--episodes-from` version of their harness (`closed_loop_eval_baselines.py`) is commit
`37852c9d`, which exists only on the yj server and was never pushed. Nothing from it is needed:
the manifest is ours.

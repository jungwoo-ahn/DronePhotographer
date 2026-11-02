# Why the shot-profile policies fail on the 3DGS benchmark (Table 5)

On `configs/eval_episodes_gs_v1.json` (90 episodes, 5 reconstructed real scenes) every
shot-profile policy scores 0-2% Succ while Image-to-action scores 57.8%. The policies keep the
distance roughly right but pick the orbit direction around the subject at chance (first chunk:
Ours 61%, pi0.5 42%; on benchmark v2 the same policies are 99-100%). The scripts here are the
probes that located the cause. Outputs live on NAS under `runs/bench_gs_probe/` (gitignored).

**Conclusion:** not a setup error. The policies do not read the goal *bearing*; they infer the
orbit direction from framing patterns that the v7 training trajectories (and benchmark v2, built
from the same trajectories) carry and the 3DGS episodes do not.

| Probe | What changes | First-chunk direction |
|---|---|---|
| 3DGS as is | — | 0.61 |
| `GS_BACKGROUND=gray` | splat background -> flat gray | 0.55 |
| `GS_SUBJECT_VIEW=AgX` | subject tone mapping Standard -> AgX (training uses Filmic/AgX) | 0.61 |
| gray + `GS_SUBJECT_LIGHT=uniform` + AgX | no background, flat light | 0.55 |
| `build_v2_on_gray.py` | **benchmark v2 episodes** through the same gray renderer | **1.00** |
| `build_v2_rotated.py 90` | same, whole episode rotated 90 deg | 1.00 (pipeline is rotation-correct) |
| `build_bearing_probe.py` | only the goal bearing changed by +-30/60 deg | follows the ORIGINAL goal 1.00, the new bearing 0.50 |
| `direction_shortcut.py` | framing-only classifier (no bearing) | train CV 0.996, v2 0.99, 3DGS 0.48 |
| `../build_spin_episodes.py --yaw 180` | v2 scenes, subject spun in place | Succ Ours 56 -> 23%, pi0.5 48 -> 13% |

Also verified identical to training: label code (240/240 frames exact), left/right (YOLO-pose
keypoints, facing offset 0 +- 5 deg), subject renders from the same relative pose, the 3DGS
geometry dumper, lens/scale/action decoding.

## Running the probes

The render switches are environment variables read by `scripts/gs_render_server.py` (background)
and `scripts/gs_subject_blender.py` (subject light / tone mapping). A probe is a one-chunk rollout:

```bash
sbatch --qos=own --export=ALL,GS_BACKGROUND=gray,GS_SUBJECT_LIGHT=uniform,GS_SUBJECT_VIEW=AgX \
    scripts/sbatch_closed_loop_eval.sh --policy cosmos \
    --checkpoint runs/train/cosmos3_camera_policy/action_sft/camera_policy_nano_v6/checkpoints/iter_000045000 \
    --episodes-from <manifest> --render-res 1024 768 --chunks 1 --stop-on-shoot 0 --seed 0 \
    --out runs/bench_gs_probe/<name>.json
```

Manifests: `configs/eval_episodes_gs_v1.json` (3DGS), or the ones these scripts write
(`runs/bench_gs_probe/{v2gray,v2gray_rot90,bearing_gs,bearing_v2}/episodes.json`).
`probe_report.py` and `bearing_report.py` print the first-chunk direction agreement;
`direction_shortcut.py` needs no GPU.

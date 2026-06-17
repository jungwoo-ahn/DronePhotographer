# AutoPhoto baseline — references & attribution

AutoPhoto baseline (issue #22): RL with an aesthetic reward. We **reuse the
pretrained aesthetic scorer** and **retrain the policy** in our Blender env.

| Source | What we took |
|---|---|
| **AutoPhoto** — Zayer et al., "AutoPhoto: Aesthetic Photo Capture using Reinforcement Learning", IROS 2021 — github.com/HadiZayer/AutoPhoto | The method (RL agent maximizing an aesthetic reward via discrete camera nav), the reward model + its checkpoint, the action set, and the env/reward structure (reimplemented over Blender). |
| **Adobe `models_lpf`** (anti-aliased CNNs, CC BY-NC-SA 4.0) | The low-pass ResNet18 backbone of the reward model (`vendor/models_lpf/`). |
| **stable-baselines3 / sb3-contrib** (MIT) | RecurrentPPO (PPO + LSTM), replacing AutoPhoto's deprecated stable-baselines 2.10 / TF1.x. |

## What is vendored / reused
- `vendor/models_lpf/{__init__,resnet,downsample}.py` — minimal low-pass ResNet for
  the reward (alexnet/densenet/mobilenet/vgg dropped; `resnet.py`'s absolute
  `robust_cnns.models_lpf` import made relative).
- `reward.py` — slim reimplementation of AutoPhoto's `ScoringModel`: loads
  `resnet-model42.pt` into `resnet18(filter_size=3)` + `Linear(512,1)`, **sign-flips
  the head** (their convention: higher = better), exposes `score()` and 512-d
  `score_and_features()` (the policy observation). Checkpoint lives in gitignored
  `weights/autophoto/` (fetched from the repo, not committed).

## What is reimplemented (not ported)
- `env.py::PhotoEnv` — a `gymnasium.Env` replacing AutoPhoto's `HabitatEnv`: the same
  9 discrete actions (`fine_turns`), aesthetic-feature observation, and
  score-gradient + time-penalty + exploration reward, but driving our
  `BlenderRolloutEnv` + `AestheticReward`.
- `renderer.py::PersistentBlenderRenderer` + `scripts/blender_render_worker.py` —
  keep one Blender process alive (EEVEE + low samples) so RL rollouts are tractable.

## Faithful simplifications (vs. AutoPhoto)
- **Terminal reward**: dense `final_score - init_score` instead of their
  distance-based ±1 (which needs a per-scene sample DB we don't carry).
- **Renderer**: EEVEE + low samples for RL speed (vs. their fast Habitat sim);
  **reduced training scale** (~1e5 steps vs. their 1.5M Habitat steps).
- **RL lib**: RecurrentPPO (SB3) instead of stable-baselines 2.10 / TF1.x.

## Where it sits on the previsualization axis
The **RL "implicit previsualization"** baseline: the learned policy/value implicitly
anticipates better viewpoints, but there is no explicit world model, and it is
**goal-agnostic** (optimizes aesthetics, not a specified shot profile). Eval reports
its aesthetic score + the shared pose-proxy distance to a target (expected poor).

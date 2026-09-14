# baseline_ckpts — provenance

Copied 2026-09-14 from `yj-1:/home/nas5/jooyeolyun/repos/DronePhotographer` (jooyeolyun's
DronePhotographer repo, branch `008-baseline-retrain`, commit `37852c9d`) for V12-side
baseline comparison. Final checkpoints only — no optimizer/training state, no intermediate
steps. Source files were left in place (copy, not move).

Full context: `docs/baselines.md` in the source repo (results on the shared 64-episode
manifest `configs/eval_episodes_v1.json`, published on `DronePhotographerV12@002-goal-authoring`).

## Contents

| Path | What | Eval (shared 64-ep manifest) |
|---|---|---|
| `pi05_fair/checkpoints/300000/pretrained_model/` | pi0.5 (LeRobot `pi05_base` ~3B, flow matching), fine-tuned full, 300k steps. **Strongest baseline.** Val-best = final ckpt (recon 18.9 cm). | mean imp 0.612, median 0.535, frac+ 0.92, d_end 0.254, succ@0.10 30% |
| `groot_fair/checkpoints/210000/pretrained_model/` | GR00T N1.7-3B (LeRobot `groot`), half-LR (5e-5) retrain; run died at 210k of 300k (rank-1 SIGABRT), ckpt intact. | published row was @130k: mean imp 0.235, d_end 0.631, succ@0.10 16% (210k eval was in flight) |
| `20260818_064526_diffusion_policy_dinov2/ckpt_best.pt` | Diffusion Policy (frozen DINOv2-L + ConditionalUnet1D, DDPM), early-stopped, best val 0.0039. `config.yaml` alongside. | mean imp 0.357, d_end 0.508, succ@0.10 5% |
| `autophoto/autophoto_22000_steps.zip` | AutoPhoto RL baseline (SB3 RecurrentPPO), final ckpt @22k steps. Goal-agnostic. | not yet re-run on new substrate |
| `weights/autophoto/resnet-model42.pt` | AutoPhoto's pretrained aesthetic ResNet18 (reward model). | — |
| `weights/unic/unic_pretrained.pth` | UNIC composition model (training-free baseline). | — |

## Caveats

- **GR00T**: the published eval row used a 130k checkpoint from the *first* fair run, which
  diverged (NaN at 168k) and whose tree was deleted. 210k from the stable half-LR retrain is
  the only surviving GR00T weight — budget-unequal vs pi0.5's 300k, in GR00T's disfavour.
- pi0.5 270k (the robustness-check row) was likewise deleted at the source; only 300k shipped.
- Action space 10D `[dtrans_cam(3), rot6d(6), shoot(1)]`, goal = shot profile with
  `subject_bearing_deg`; VLAs consume the goal as NL prompt (`goal_text.goal_prompt`), DP as a
  numeric vector. Eval protocol must use `--episodes-from configs/eval_episodes_v1.json`.
- Per-family training losses are not comparable; rank with closed-loop rollout, not recon.

## sha256 (verified after transfer, local == remote)

```
258d39112f48fdc4e57d7b35eea3963cf407112ad15c3788cf83b30b86141afe  pi05_fair/checkpoints/300000/pretrained_model/model.safetensors
00ec4e272988ffd379f65a9f6fd1214e9ce6dcfa93e768ff3e90270203e2cf1e  pi05_fair/checkpoints/300000/pretrained_model/policy_preprocessor_step_3_normalizer_processor.safetensors
cdc92c2610f998ef9532c085c73fb2fe6ad62b2627bef5346c2bfa3debe1496b  pi05_fair/checkpoints/300000/pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors
abdef5586a4cc7f882a98683a6995c7e8d874d88e79f899d2c58a0e27a0e9c24  groot_fair/checkpoints/210000/pretrained_model/model.safetensors
c8958c341bced2e212179964c3bbb02cf60eea3814c27fbc3d7f63f143545df5  groot_fair/checkpoints/210000/pretrained_model/policy_preprocessor_step_2_groot_n1_7_pack_inputs_v1.safetensors
c8958c341bced2e212179964c3bbb02cf60eea3814c27fbc3d7f63f143545df5  groot_fair/checkpoints/210000/pretrained_model/policy_postprocessor_step_0_groot_action_unpack_unnormalize_v2.safetensors
e9b714787158222a4b9c4e2f2e9cc6ba0e05c1ecfb3f054b481117a1a86c7d8a  20260818_064526_diffusion_policy_dinov2/ckpt_best.pt
9a7f3f16ec7816bcb3d3ee3dc1b1bd2df86fa991cb28b9c5e94f69b4469d86df  autophoto/autophoto_22000_steps.zip
82b664b63882bc101ca762412023b4825b256f26c8c4e80b0f7c42edb4d31479  weights/autophoto/resnet-model42.pt
8f9f4cc5f4b7663f1f47f4c955492b408b1c2deeeaddee2f4fac71da764d913e  weights/unic/unic_pretrained.pth
```
(GR00T의 pre/postprocessor 두 파일은 내용이 byte-identical — 해시 동일이 정상.)

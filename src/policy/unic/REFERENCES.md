# UNIC baseline — references & attribution

UNIC baseline (issue #22): an aesthetic/composition model used **reactively** as a
camera policy. We use the released pretrained model **as-is** (no training).

| Source | What we took |
|---|---|
| **UNIC** — Zhang et al., "Beyond Image Borders: Learning Feature Extrapolation for Unbounded Image Composition", ICCV 2023 — github.com/liuxiaoyu1104/UNIC | The model + pretrained checkpoint. It recommends a composition bounding box for the current view (possibly extending beyond the borders). |
| **Conditional DETR** (Microsoft, Apache-2.0) / **DETR** (Facebook, Apache-2.0) | UNIC's detection backbone, vendored transitively. |

## What is vendored
`vendor/` holds the minimal **inference-only** subset of the UNIC repo (commit on
`main`, fetched June 2026):

```
vendor/models/{__init__,conditional_detr,backbone,transformer,attention,position_encoding,QEM}.py
vendor/util/{misc,box_ops}.py
```

Excluded (training-only): `matcher.py` (Hungarian loss), `segmentation.py`
(panoptic; pulls `panopticapi`), datasets, engine, main. Only `torch` +
`torchvision` are needed on the inference path (no compiled CUDA ops — `QEM.py`
uses `torchvision.ops.DeformConv2d`).

### Edits to vendored files (all marked `VENDORED ...` in-file)
- `models/conditional_detr.py`: commented the top-level `from .matcher` and
  `from .segmentation` imports (used only by the training `build()` / `SetCriterion`
  paths we never call) and made `import cv2` optional.
- `models/attention.py`: fixed two upstream version guards that mis-parse on
  torch>=2 (operator precedence) and imported the removed `_LinearWithBias` /
  `torch._overrides`; now prefer `NonDynamicallyQuantizableLinear` and
  `torch.overrides`.
- All `from util.*` / `from models.*` imports rewritten to package-relative
  (`from ..util.*`, `from .*`).

We construct the model directly (backbone + transformer + ConditionalDETR +
PostProcess) from the **exact `args` stored in the checkpoint**, loading the **EMA**
weights (`ema_model.*`). See `model.py`.

## Checkpoint
Pretrained `.pth` (~926 MB) from the UNIC repo's Google Drive, downloaded to
`weights/unic/unic_pretrained.pth` (the `weights/` dir is gitignored — fetch at
setup, never commit). The checkpoint stores `model`, `ema`, `args`, etc.

## Reactive policy mapping (`policy.py`)
UNIC's recommended box → a camera-local 5D action: box-center offset from the frame
center → **pan** (yaw/pitch via the camera FOV); recommended box width vs the frame
→ **zoom** (dolly `dz`). Lateral translation is left at zero (pan is rotation;
depth is unknown so the dolly is a monotonic heuristic).

## Where it sits on the previsualization axis
The **"reactive, no previsualization"** baseline: UNIC scores/recommends only from
the current frame and is goal-agnostic (no notion of the target shot profile). Same
target spec / 5D action / pose-proxy eval as the other baselines for comparability.

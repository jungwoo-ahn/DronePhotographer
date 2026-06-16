"""Load + run the vendored UNIC composition model for inference.

UNIC ("Beyond Image Borders", ICCV 2023) recommends a composition bounding box for
the current view — possibly extending *beyond* the image borders (its unbounded /
feature-extrapolation contribution). We use the released pretrained model as-is
(no training) and read out the top recommended box; `policy.py` turns that box into
a camera move. See REFERENCES.md for the vendoring details.

Construction mirrors the upstream `build()` (backbone + transformer +
ConditionalDETR + PostProcess), skipping the training-only criterion/matcher, and
reads the exact architecture `args` stored in the checkpoint so it always matches
the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.policy.unic.vendor.models.backbone import build_backbone
from src.policy.unic.vendor.models.conditional_detr import ConditionalDETR, PostProcess
from src.policy.unic.vendor.models.transformer import build_transformer
from src.policy.unic.vendor.util.misc import nested_tensor_from_tensor_list

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_SHORT = 864          # UNIC eval transform: shorter edge -> 864, keep aspect


def _ema_state_dict(ck: dict) -> dict:
    """Extract the EMA weights (`ema_model.*`) from an ema_pytorch checkpoint dict."""
    e = ck["ema"]
    pref = "ema_model."
    return {k[len(pref):]: v for k, v in e.items() if k.startswith(pref)}


@dataclass
class UNICRecommendation:
    """Top recommended composition box, in normalized [0,1] image coords.

    Coords may fall outside [0,1] (UNIC's unbounded composition). `center_x/y` is
    the box center; `width/height` are box extents as fractions of the frame.
    """

    center_x: float
    center_y: float
    width: float
    height: float
    score: float


class UNICModel:
    def __init__(self, model: torch.nn.Module, postprocess: PostProcess, device: str) -> None:
        self.model = model
        self.postprocess = postprocess
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str | Path, *, device: str = "cuda", use_ema: bool = True) -> "UNICModel":
        ck = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        args = ck["args"]
        args.device = device
        num_classes = 250 if getattr(args, "dataset_file", "coco") == "coco_panoptic" else 1
        backbone = build_backbone(args)
        transformer = build_transformer(args)
        model = ConditionalDETR(backbone, transformer, num_classes=num_classes,
                                num_queries=args.num_queries, aux_loss=args.aux_loss)
        state = _ema_state_dict(ck) if use_ema else ck["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[UNIC] load_state_dict: missing={len(missing)} unexpected={len(unexpected)} "
                  f"(ema={use_ema})")
        model.eval().to(device)
        return cls(model, PostProcess(), device)

    def _preprocess(self, image):
        """PIL.Image -> (NestedTensor, (orig_h, orig_w))."""
        import torchvision.transforms.functional as TF

        w0, h0 = image.size
        short = min(h0, w0)
        if short != RESIZE_SHORT:
            r = RESIZE_SHORT / short
            image = TF.resize(image, [round(h0 * r), round(w0 * r)])
        t = TF.to_tensor(image.convert("RGB"))
        t = TF.normalize(t, list(IMAGENET_MEAN), list(IMAGENET_STD))
        samples = nested_tensor_from_tensor_list([t]).to(self.device)
        return samples, (h0, w0)

    @torch.no_grad()
    def recommend(self, image) -> UNICRecommendation:
        """Run UNIC on a PIL image and return the top recommended composition box."""
        samples, (h0, w0) = self._preprocess(image)
        outputs = self.model(samples, True, "soft")
        sizes = torch.tensor([[h0, w0]], device=self.device, dtype=torch.float32)
        start = torch.zeros((1, 4), device=self.device, dtype=torch.float32)  # full frame, no crop offset
        results = self.postprocess(outputs, sizes, start)
        r = results[0]
        i = int(torch.argmax(r["scores"]))
        x1, y1, x2, y2 = (float(v) for v in r["boxes"][i].cpu().numpy())
        # normalize to fractions of the original frame (may be <0 or >1 — unbounded)
        cx = ((x1 + x2) / 2.0) / w0
        cy = ((y1 + y2) / 2.0) / h0
        bw = (x2 - x1) / w0
        bh = (y2 - y1) / h0
        return UNICRecommendation(cx, cy, abs(bw), abs(bh), float(r["scores"][i]))


__all__ = ["UNICModel", "UNICRecommendation", "IMAGENET_MEAN", "IMAGENET_STD", "RESIZE_SHORT"]

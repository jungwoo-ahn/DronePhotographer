"""Does the frame the policy stopped on LOOK like the frame that was asked for?

`closed_loop_eval.py` scores geometry: where the camera ended up relative to the goal pose,
in radians. That is exact and policy-independent, which is why it is the ranking metric — but
it is a statement about the camera, not about the picture. Two poses a tenth of a radian apart
can differ by an occluding pillar, and the geometry cannot say so.

This reads the frames those rollouts already saved and compares each policy's FINAL frame with
the goal frame, on four metrics of increasing semantic level:

    ssim    structural similarity on luma -- pixels, no learning
    vgg     cosine on VGG16 relu4_3 features -- the classic perceptual backbone
    clip    cosine on CLIP ViT-L/14 image embeddings -- semantic, text-aligned
    dinov3  cosine on DINOv3 ViT-B/16 -- self-supervised, strongest at spatial layout

Nothing here needs a GPU rollout: the rollouts are already on disk (`shots[-1].path`), and the
goal frames come from the manifest. Changing or adding a metric is a re-read, not a re-run.

TWO THINGS MAKE THE NUMBERS READABLE, and both matter more than the raw similarity:

1. **The start frame is scored too.** A similarity of 0.62 means nothing alone; 0.41 -> 0.62
   against a goal means the policy moved the picture two thirds of the way there. Same shape as
   `improvement = d_start - d_end`, and a no-op scores exactly 0 here as well.
2. **A floor AND a ceiling are measured, not assumed.** The floor is the goal frame against
   frames from other placements -- what "unrelated picture" scores. The ceiling is the part that
   is easy to forget and changes every conclusion: the goal frame was rendered by the TRAINING
   pipeline and the rollout frame by OURS, so even a policy that ends exactly on the goal pose
   does not score 1.0. It is measured on the START frame, which both pipelines rendered from the
   identical pose, and it is low -- DINOv3 0.865, SSIM 0.541. Normalising to 1.0 instead makes a
   policy at 98% of what is achievable look like it is at 65% of the way there.

    .venv-vla/bin/python scripts/perceptual_eval.py --runs runs/closed_loop/cmp_*.json
venv: .venv-vla (needs torch + timm + transformers; CPU is fine, ~2 min a policy on GPU).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

V12 = "/home/nas_main/jungwooahn/projects/DronePhotographerV12"
sys.path.insert(0, V12)

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="cmp_<policy>.json from closed_loop_eval")
ap.add_argument("--manifest", default=f"{V12}/configs/eval_episodes_v1.json")
ap.add_argument("--metrics", default="ssim,vgg,clip,dinov3")
ap.add_argument("--device", default="cuda")
ap.add_argument("--floor-samples", type=int, default=3,
                help="unrelated goal frames per episode, to measure what 'unrelated' scores")
ap.add_argument("--out", default=f"{V12}/runs/eval/perceptual.json")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()


# ---------------------------------------------------------------- metrics
class Metric:
    """Higher is always MORE SIMILAR, so every metric points the same way as the table reads."""

    name: str

    def __call__(self, a: Path, b: Path) -> float: ...


class SSIM(Metric):
    name = "ssim"

    def __init__(self, device):
        from skimage.metrics import structural_similarity
        self._f = structural_similarity

    def __call__(self, a, b):
        from PIL import Image
        ia = np.asarray(Image.open(a).convert("L"), dtype=np.float32)
        ib = np.asarray(Image.open(b).convert("L"), dtype=np.float32)
        if ia.shape != ib.shape:
            ib = np.asarray(Image.open(b).convert("L").resize(ia.shape[::-1]), dtype=np.float32)
        return float(self._f(ia, ib, data_range=255.0))


class _Embedding(Metric):
    """Cosine between two image embeddings. Subclasses supply the encoder and its preprocessing."""

    def __init__(self, device):
        import torch
        self._torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._cache: dict[str, "torch.Tensor"] = {}

    def embed(self, path: Path): ...

    def __call__(self, a, b):
        torch = self._torch
        ea, eb = self._cached(str(a)), self._cached(str(b))
        return float(torch.nn.functional.cosine_similarity(ea, eb, dim=-1).item())

    def _cached(self, p: str):
        # A goal frame is compared with every policy's end frame and with the floor samples,
        # so caching turns an O(policies x episodes) encode into O(distinct frames).
        if p not in self._cache:
            self._cache[p] = self.embed(Path(p))
        return self._cache[p]


class VGG(_Embedding):
    """VGG16 relu4_3, the layer LPIPS weights most heavily. Global-average-pooled, then cosine.

    Not LPIPS itself: LPIPS needs its learned per-channel weights and is a DISTANCE on aligned
    images. These frames are two different camera poses of one scene, so a spatially-aligned
    distance would mostly report "the camera moved", which is what the geometry already says.
    """

    name = "vgg"

    def __init__(self, device):
        super().__init__(device)
        import torch
        from torchvision.models import VGG16_Weights, vgg16
        w = VGG16_Weights.IMAGENET1K_V1
        self.tf = w.transforms()
        net = vgg16(weights=w).features[:23].eval().to(self.device)   # through relu4_3
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net

    def embed(self, path):
        import torch
        from PIL import Image
        x = self.tf(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.net(x)
        return f.mean(dim=(2, 3)).flatten()


class CLIP(_Embedding):
    name = "clip"

    def __init__(self, device):
        super().__init__(device)
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        rid = "openai/clip-vit-large-patch14"
        self.proc = CLIPImageProcessor.from_pretrained(rid)
        self.net = CLIPVisionModelWithProjection.from_pretrained(rid).eval().to(self.device)

    def embed(self, path):
        import torch
        from PIL import Image
        x = self.proc(images=Image.open(path).convert("RGB"), return_tensors="pt")
        x = {k: v.to(self.device) for k, v in x.items()}
        with torch.no_grad():
            return self.net(**x).image_embeds.flatten()


class DINOv3(_Embedding):
    name = "dinov3"

    def __init__(self, device):
        super().__init__(device)
        import timm
        import torch
        self.net = timm.create_model("vit_base_patch16_dinov3.lvd1689m",
                                     pretrained=True, num_classes=0).eval().to(self.device)
        cfg = timm.data.resolve_model_data_config(self.net)
        self.tf = timm.data.create_transform(**cfg, is_training=False)

    def embed(self, path):
        import torch
        from PIL import Image
        x = self.tf(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.net(x).flatten()


BUILDERS = {"ssim": SSIM, "vgg": VGG, "clip": CLIP, "dinov3": DINOv3}


# ---------------------------------------------------------------- data
def load_runs(paths):
    out = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        name = d["summary"].get("policy_label") or Path(p).stem.replace("cmp_", "")
        out[name] = {e["key"]: e for e in d["episodes"]}
    return out


def main() -> int:
    rng = random.Random(args.seed)
    manifest = {e["key"]: e for e in json.loads(Path(args.manifest).read_text())["episodes"]}
    runs = load_runs(args.runs)
    shared = sorted(set.intersection(*(set(v) for v in runs.values())))
    print(f"{len(runs)} policies, {len(shared)} shared episodes", flush=True)

    goals = {k: manifest[k]["goal_image"] for k in shared if os.path.exists(manifest[k]["goal_image"])}
    starts = {k: manifest[k]["start_image"] for k in shared
              if os.path.exists(manifest[k].get("start_image", ""))}
    missing = [k for k in shared if k not in goals]
    if missing:
        print(f"  {len(missing)} episodes have no goal image on disk; skipped", flush=True)
    keys = [k for k in shared if k in goals]

    results: dict[str, dict] = {}
    for mname in [m.strip() for m in args.metrics.split(",") if m.strip()]:
        metric = BUILDERS[mname](args.device)
        print(f"\n=== {mname} ===", flush=True)
        per_policy: dict[str, dict[str, float]] = {}

        # floor: the goal frame against unrelated goals. What "a different picture" scores.
        floor = []
        for k in keys:
            others = rng.sample([o for o in keys if o != k], min(args.floor_samples, len(keys) - 1))
            floor += [metric(Path(goals[k]), Path(goals[o])) for o in others]
        floor_mu = float(np.mean(floor))

        # ceiling: the SAME pose rendered by both pipelines. The rollout's start frame is ours;
        # the manifest's start_image is the training render of that identical pose. Whatever
        # separates them is renderer, not policy, and no policy can score above it.
        ref = next(iter(runs.values()))
        ceil_vals = [metric(Path(ref[k]["shots"][0]["path"]), Path(starts[k]))
                     for k in keys if k in starts]
        ceil_mu = float(np.mean(ceil_vals)) if ceil_vals else 1.0

        for pol, eps in runs.items():
            s_start, s_end = [], []
            for k in keys:
                e = eps[k]
                s_start.append(metric(Path(e["shots"][0]["path"]), Path(goals[k])))
                s_end.append(metric(Path(e["shots"][-1]["path"]), Path(goals[k])))
            s_start, s_end = np.array(s_start), np.array(s_end)
            gain = s_end - s_start
            # How far along the way from "unrelated" to "the goal itself" the final frame is.
            # 1.0 would be the goal frame; 0.0 is an unrelated picture.
            span = max(1e-9, ceil_mu - floor_mu)
            per_policy[pol] = {
                "start": float(s_start.mean()), "end": float(s_end.mean()),
                "gain": float(gain.mean()), "gain_median": float(np.median(gain)),
                "frac_positive": float((gain > 0).mean()),
                # 1.0 = as close as this renderer pair allows, not as close as 1.0 would be
                "toward_goal": float((s_end.mean() - floor_mu) / span),
            }
            print(f"  {pol:10} start {s_start.mean():.4f} -> end {s_end.mean():.4f}  "
                  f"gain {gain.mean():+.4f} (median {np.median(gain):+.4f}, "
                  f"{(gain > 0).mean():.0%} positive)  toward_goal {per_policy[pol]['toward_goal']:.2f}",
                  flush=True)
        print(f"  {'FLOOR':10} unrelated goal frames         {floor_mu:.4f}  (n={len(floor)})", flush=True)
        print(f"  {'CEILING':10} same pose, the two renderers  {ceil_mu:.4f}  (n={len(ceil_vals)})"
              f"   <- what a perfect policy scores", flush=True)

        # correlation with the geometric distance: if it tracks `d` perfectly it adds nothing
        any_pol = next(iter(runs))
        d_end = np.array([runs[any_pol][k]["d_end"] for k in keys])
        pe = np.array([metric(Path(runs[any_pol][k]["shots"][-1]["path"]), Path(goals[k])) for k in keys])
        r = float(np.corrcoef(d_end, pe)[0, 1])
        print(f"  corr(d_end, {mname}) on {any_pol}: {r:+.3f}   "
              f"({'tracks the geometry' if abs(r) > 0.7 else 'says something the geometry does not'})",
              flush=True)
        results[mname] = {"per_policy": per_policy, "floor": floor_mu, "ceiling": ceil_mu,
                          "corr_d_end": r, "corr_ref_policy": any_pol}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "_what": "Perceptual similarity between each policy's FINAL frame and the goal frame. "
                 "Higher is more similar for every metric. `gain` is end - start, so a no-op "
                 "scores 0; `toward_goal` is where the final frame sits between an unrelated "
                 "picture (0) and the CEILING (1) -- the ceiling being the same pose rendered "
                 "by the training pipeline and by ours, which is all a perfect policy can reach.",
        "episodes": len(keys), "manifest": args.manifest, "runs": list(runs),
        "metrics": results}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

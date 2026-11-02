"""Shared splat I/O and rendering for the subject-compositing tools.

Conventions: splat world in its own units; `mpu` = metres per unit; `up` = unit up vector in
splat coordinates; cameras are 4x4 camera-to-world in OpenCV axes (x right, y down, z forward).
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from read_write_model import qvec2rotmat, read_cameras_binary, read_images_binary  # noqa: E402

DEV = torch.device("cuda")


class Splat:
    def __init__(self, ply: str | Path):
        v = PlyData.read(str(ply))["vertex"].data
        nm = v.dtype.names
        self.xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
        rest_n = sorted([n for n in nm if n.startswith("f_rest_")], key=lambda s: int(s.split("_")[-1]))
        dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], -1)[:, None]
        rest = (np.stack([v[n] for n in rest_n], -1).reshape(len(self.xyz), 3, -1).transpose(0, 2, 1)
                if rest_n else np.zeros((len(self.xyz), 0, 3)))
        sh = np.concatenate([dc, rest], 1).astype(np.float32)
        self.deg = int(round(math.sqrt(sh.shape[1]))) - 1
        T = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(DEV)
        self.opacity = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float32)))
        self.t_means = T(self.xyz)
        self.t_quats = torch.nn.functional.normalize(T(np.stack([v[f"rot_{i}"] for i in range(4)], -1)), dim=-1)
        self.t_scales = torch.exp(T(np.stack([v[f"scale_{i}"] for i in range(3)], -1)))
        self.t_opac = T(self.opacity)
        self.t_sh = T(sh)

    @torch.no_grad()
    def render(self, c2w: np.ndarray, K: np.ndarray, w: int, h: int, near: float = 0.01, far: float = 1e10):
        """-> rgb uint8 (h,w,3) over black, planar depth (units), alpha. Gaussians whose centres lie
        beyond `far` are left out, so alpha at a pixel can measure what stands in front of a point."""
        from gsplat import rasterization
        vm = torch.from_numpy(np.linalg.inv(c2w)).float().to(DEV)[None]
        Kt = torch.from_numpy(np.asarray(K)).float().to(DEV)[None]
        out, alpha, _ = rasterization(self.t_means, self.t_quats, self.t_scales, self.t_opac, self.t_sh,
                                      vm, Kt, w, h, sh_degree=self.deg, near_plane=near,
                                      far_plane=far, render_mode="RGB+ED")
        rgb = (out[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        return rgb, out[0, ..., 3].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()


def load_cameras(kind: str, path: str | Path):
    """-> list of (name, c2w 4x4 OpenCV, K 3x3, (w, h))."""
    path = Path(path)
    cams = []
    if kind == "inria":
        for c in json.load(open(path)):
            c2w = np.eye(4); c2w[:3, :3] = np.array(c["rotation"]); c2w[:3, 3] = c["position"]
            K = np.array([[c["fx"], 0, c["width"] / 2], [0, c["fy"], c["height"] / 2], [0, 0, 1]])
            cams.append((c["img_name"], c2w, K, (c["width"], c["height"])))
    elif kind == "nb":
        for f in sorted(path.glob("*.npz")):
            d = np.load(f)
            c2w = np.eye(4); c2w[:3, :4] = d["poses"]
            fx, fy, cx, cy = d["intrinsics"]
            cams.append((f.stem, c2w, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), tuple(d["image_sizes"])))
    elif kind == "colmap":
        cm = read_cameras_binary(path / "cameras.bin")
        for im in read_images_binary(path / "images.bin").values():
            c = cm[im.camera_id]
            if c.model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE", "RADIAL"):
                fx = fy = c.params[0]; cx, cy = c.params[1], c.params[2]
            else:
                fx, fy, cx, cy = c.params[:4]
            w2c = np.eye(4); w2c[:3, :3] = qvec2rotmat(im.qvec); w2c[:3, 3] = im.tvec
            cams.append((im.name, np.linalg.inv(w2c), np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                         (c.width, c.height)))
    else:
        raise ValueError(kind)
    return cams


def K_hfov(hfov_deg: float, w: int, h: int) -> np.ndarray:
    f = w / (2 * math.tan(math.radians(hfov_deg) / 2))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye; f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, np.array([1.0, 0, 0]))
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    c2w = np.eye(4); c2w[:3, 0] = r; c2w[:3, 1] = d; c2w[:3, 2] = f; c2w[:3, 3] = eye
    return c2w


def draw_spots(img, c2w: np.ndarray, K: np.ndarray, marks) -> None:
    """Draw (point, label, colour) marks on an image rendered with camera (c2w, K)."""
    from PIL import ImageDraw
    w2c = np.linalg.inv(c2w)[:3]
    dr = ImageDraw.Draw(img)
    for P, label, colour in marks:
        pc = w2c @ np.append(P, 1)
        if pc[2] <= 0:
            continue
        px = K @ pc; px = px[:2] / px[2]
        dr.ellipse([px[0] - 9, px[1] - 9, px[0] + 9, px[1] + 9], outline=colour, width=3)
        dr.text((px[0] + 11, px[1] - 8), label, fill=(255, 255, 0))

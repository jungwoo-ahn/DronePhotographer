from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .prompt import build_action_text
from .rotation_utils import (
    batch_relative_rotation_angle_deg,
    relative_rotation_rotvec,
    relative_rotation_rotvec_camera_local,
    relative_translation_camera_local,
    target_orientation_forward_up_camera_local,
    target_orientation_forward_up_world,
)
from .schema import SCORE_KEYS, extract_scores_from_annotation, scores_to_canonical_json

_VALID_DISTRIBUTIONS = {"natural", "uniform", "log_uniform", "adaptive"}


@dataclass(frozen=True)
class PairRecord:
    index_i: int
    index_j: int
    image_i: Path
    action_text: str
    target_text: str
    target_scores: dict[str, float]
    distance: float = 0.0
    bucket_idx: int = 0


@dataclass(frozen=True)
class ViewRecord:
    image_path: Path
    camera_position: np.ndarray
    camera_forward: np.ndarray
    camera_up: np.ndarray
    has_detection: bool
    scores: dict[str, float]
    placement_id: str = "default"


class DroneActionScoreDataset(Dataset):
    def __init__(
        self,
        annotations_path: str | Path,
        image_root: str | Path | None = None,
        action_frame: str = "camera_local",
        rotation_representation: str = "orientation_6d",
        distance_threshold: float = 1.5,
        rotation_threshold_deg: float | None = 60.0,
        pair_distance_distribution: str = "log_uniform",
        n_distance_bins: int = 5,
        min_pair_distance: float = 0.05,
        max_pairs_per_image: int = 32,
        zero_action_ratio: float = 0.0,
        seed: int = 721,
        target_score_keys: Sequence[str] | None = None,
        placement_start_idx: int = 0,
        placement_end_idx: int | None = None,
        views_cache_dir: str | Path | None = None,
    ) -> None:
        self.annotations_path = Path(annotations_path)
        self.image_root = Path(image_root) if image_root is not None else self.annotations_path.parent
        self.action_frame = str(action_frame)
        self.rotation_representation = str(rotation_representation)
        self.distance_threshold = float(distance_threshold)
        self.rotation_threshold_deg = None if rotation_threshold_deg is None else float(rotation_threshold_deg)
        self.pair_distance_distribution = str(pair_distance_distribution)
        self.n_distance_bins = int(n_distance_bins)
        self.min_pair_distance = float(min_pair_distance)
        self.max_pairs_per_image = int(max_pairs_per_image)
        self.zero_action_ratio = float(zero_action_ratio)
        self.seed = int(seed)
        self.target_score_keys = list(SCORE_KEYS if target_score_keys is None else target_score_keys)
        self.placement_start_idx = int(placement_start_idx)
        self.placement_end_idx = None if placement_end_idx is None else int(placement_end_idx)
        self.views_cache_dir = Path(views_cache_dir) if views_cache_dir is not None else None
        if self.placement_start_idx < 0:
            raise ValueError("placement_start_idx must be >= 0")
        if self.placement_end_idx is not None and self.placement_end_idx <= self.placement_start_idx:
            raise ValueError("placement_end_idx must be > placement_start_idx")
        if self.action_frame not in {"camera_local", "world"}:
            raise ValueError("action_frame must be 'camera_local' or 'world'")
        if self.rotation_representation not in {"orientation_6d", "rotvec"}:
            raise ValueError("rotation_representation must be 'orientation_6d' or 'rotvec'")
        if self.zero_action_ratio < 0.0 or self.zero_action_ratio >= 1.0:
            raise ValueError("zero_action_ratio must be in [0.0, 1.0)")
        if self.pair_distance_distribution not in _VALID_DISTRIBUTIONS:
            raise ValueError(
                f"pair_distance_distribution must be one of {sorted(_VALID_DISTRIBUTIONS)}; "
                f"got {self.pair_distance_distribution!r}"
            )
        if self.pair_distance_distribution != "natural" and self.n_distance_bins < 1:
            raise ValueError("n_distance_bins must be >= 1 when binning is enabled")
        if self.pair_distance_distribution == "log_uniform" and self.min_pair_distance <= 0:
            raise ValueError("min_pair_distance must be > 0 for log_uniform binning")
        if self.rotation_threshold_deg is not None and self.rotation_threshold_deg <= 0:
            raise ValueError("rotation_threshold_deg must be > 0 when set")

        cache_path = self._views_cache_path()
        if cache_path is not None and cache_path.exists():
            print(f"[dataset] loading cached views: {cache_path}", flush=True)
            t0 = time.time()
            with cache_path.open("rb") as f:
                self.views = pickle.load(f)
            print(f"[dataset] cached views loaded in {time.time()-t0:.1f}s "
                  f"(n_views={len(self.views)})", flush=True)
        else:
            raw = self._load_raw_annotations()
            self.views = self._load_views(raw)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
                with tmp.open("wb") as f:
                    pickle.dump(self.views, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, cache_path)
                print(f"[dataset] views cached to {cache_path}", flush=True)
        normal_pairs = self._build_pairs()
        zero_pairs = self._build_zero_action_pairs(base_pair_count=len(normal_pairs))
        self.pairs = normal_pairs + zero_pairs
        self.zero_action_pairs_count = len(zero_pairs)

    def _load_raw_annotations(self) -> list[dict]:
        """Load annotation entries from a single file or a directory of placements.

        Directory layout (v5): <annotations_path>/p*/annotations.json. Each entry is
        tagged with `_placement_id` (parent dir name) and `_image_root` (per-placement)
        so multi-file loading collapses to the same per-view path resolution as
        single-file loading.
        """
        if self.annotations_path.is_dir():
            ann_files = list(self.annotations_path.glob("p*/annotations.json"))
            if not ann_files:
                raise FileNotFoundError(
                    f"No placement annotations.json found under {self.annotations_path} "
                    "(expected layout: <dir>/p*/annotations.json)"
                )

            def _placement_sort_key(path: Path) -> tuple[int, str]:
                m = re.match(r"p(\d+)_", path.parent.name)
                return (int(m.group(1)) if m else 10**9, path.parent.name)

            ann_files.sort(key=_placement_sort_key)
            ann_files = ann_files[self.placement_start_idx : self.placement_end_idx]
            if not ann_files:
                raise ValueError(
                    f"placement slice [{self.placement_start_idx}:{self.placement_end_idx}] is empty"
                )
            raw: list[dict] = []
            for ann_file in ann_files:
                placement_dir = ann_file.parent
                placement_id = placement_dir.name
                with ann_file.open("r", encoding="utf-8") as f:
                    items = json.load(f)
                for item in items:
                    item["_placement_id"] = placement_id
                    item["_image_root"] = str(placement_dir)
                raw.extend(items)
            return raw

        with self.annotations_path.open("r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            item.setdefault("_placement_id", "default")
            item.setdefault("_image_root", str(self.image_root))
        return items

    def _views_cache_path(self) -> Path | None:
        """Path to a pickle cache of the views list. None if caching is disabled."""
        if self.views_cache_dir is None:
            return None
        key_parts = [
            str(self.annotations_path.resolve()),
            str(self.placement_start_idx),
            str(self.placement_end_idx),
            ",".join(self.target_score_keys),
        ]
        if self.annotations_path.is_dir():
            flag = self.annotations_path / "_cam_to_obj_convention_v2.flag"
            key_parts.append(
                hashlib.md5(flag.read_bytes()).hexdigest() if flag.exists() else "no_flag"
            )
        key = hashlib.sha1("|".join(key_parts).encode("utf-8")).hexdigest()[:16]
        return self.views_cache_dir / f"views_{key}.pkl"

    def _load_views(self, raw: list[dict]) -> list[ViewRecord]:
        views: list[ViewRecord] = []
        # Cache image dimensions per image_root: all v5 placements render at the
        # same resolution, so reading dims from the first image of each placement
        # avoids one PIL Image.open per view (320 vs 640k = ~2000x speedup).
        dims_cache: dict[str, tuple[int, int]] = {}
        t0 = time.time()
        for idx, item in enumerate(raw):
            placement_id = str(item.get("_placement_id", "default"))
            image_root_str = str(item.get("_image_root", str(self.image_root)))
            image_root = Path(image_root_str)
            image_path = image_root / str(item["image"])
            dims = dims_cache.get(image_root_str)
            if dims is None:
                if not image_path.exists():
                    raise FileNotFoundError(f"image missing: {image_path}")
                with Image.open(image_path) as image:
                    dims = image.size
                dims_cache[image_root_str] = dims
            image_width, image_height = dims

            scores = extract_scores_from_annotation(
                annotation=item,
                image_width=image_width,
                image_height=image_height,
                score_keys=self.target_score_keys,
            )

            views.append(
                ViewRecord(
                    image_path=image_path,
                    camera_position=np.asarray(item["camera_position"], dtype=np.float32),
                    camera_forward=np.asarray(
                        item.get("final_forward", item.get("base_forward")),
                        dtype=np.float32,
                    ),
                    camera_up=np.asarray(
                        item.get("final_up", item.get("base_up")),
                        dtype=np.float32,
                    ),
                    has_detection=bool(item.get("detections")),
                    scores=scores,
                    placement_id=placement_id,
                )
            )
        print(f"[dataset] loaded {len(views)} views from {len(dims_cache)} placements "
              f"in {time.time()-t0:.1f}s", flush=True)
        return views

    def _build_action_text_for_views(self, view_i: ViewRecord, view_j: ViewRecord) -> str:
        if self.action_frame == "camera_local":
            delta_position_np = relative_translation_camera_local(
                position_i=view_i.camera_position,
                position_j=view_j.camera_position,
                forward_i=view_i.camera_forward,
                up_i=view_i.camera_up,
            )
            if self.rotation_representation == "orientation_6d":
                target_forward_np, target_up_np = target_orientation_forward_up_camera_local(
                    view_i.camera_forward,
                    view_i.camera_up,
                    view_j.camera_forward,
                    view_j.camera_up,
                )
                return build_action_text(
                    delta_position=tuple(delta_position_np.tolist()),
                    action_frame=self.action_frame,
                    rotation_representation=self.rotation_representation,
                    target_forward=tuple(target_forward_np.tolist()),
                    target_up=tuple(target_up_np.tolist()),
                )
            delta_rotation_np = relative_rotation_rotvec_camera_local(
                view_i.camera_forward,
                view_i.camera_up,
                view_j.camera_forward,
                view_j.camera_up,
            )
            return build_action_text(
                delta_position=tuple(delta_position_np.tolist()),
                action_frame=self.action_frame,
                rotation_representation=self.rotation_representation,
                delta_rotation=tuple(delta_rotation_np.tolist()),
            )

        delta_position_np = view_j.camera_position - view_i.camera_position
        if self.rotation_representation == "orientation_6d":
            target_forward_np, target_up_np = target_orientation_forward_up_world(
                view_j.camera_forward,
                view_j.camera_up,
            )
            return build_action_text(
                delta_position=tuple(delta_position_np.tolist()),
                action_frame=self.action_frame,
                rotation_representation=self.rotation_representation,
                target_forward=tuple(target_forward_np.tolist()),
                target_up=tuple(target_up_np.tolist()),
            )
        delta_rotation_np = relative_rotation_rotvec(
            view_i.camera_forward,
            view_i.camera_up,
            view_j.camera_forward,
            view_j.camera_up,
        )
        return build_action_text(
            delta_position=tuple(delta_position_np.tolist()),
            action_frame=self.action_frame,
            rotation_representation=self.rotation_representation,
            delta_rotation=tuple(delta_rotation_np.tolist()),
        )

    def _compute_distance_bin_edges(self) -> np.ndarray | None:
        """Bin edges (length n+1) for distance binning. None if natural distribution."""
        if self.pair_distance_distribution == "log_uniform":
            return np.exp(
                np.linspace(
                    np.log(self.min_pair_distance),
                    np.log(self.distance_threshold),
                    self.n_distance_bins + 1,
                )
            )
        if self.pair_distance_distribution in {"uniform", "adaptive"}:
            return np.linspace(0.0, self.distance_threshold, self.n_distance_bins + 1)
        return None  # natural

    def _build_pairs(self) -> list[PairRecord]:
        rng = np.random.default_rng(self.seed)
        pair_records: list[PairRecord] = []
        n = len(self.views)
        if n == 0:
            return pair_records

        positions_all = np.stack([v.camera_position for v in self.views], axis=0)
        forwards_all = np.stack([v.camera_forward for v in self.views], axis=0)
        ups_all = np.stack([v.camera_up for v in self.views], axis=0)

        # Group view indices by placement_id so pairs only form within a placement.
        groups: dict[str, list[int]] = defaultdict(list)
        for k, view in enumerate(self.views):
            groups[view.placement_id].append(k)

        bin_edges = self._compute_distance_bin_edges()
        target_per_bin = max(1, self.max_pairs_per_image // self.n_distance_bins)

        for placement_id, gidxs in groups.items():
            gidx_arr = np.asarray(gidxs, dtype=np.int64)
            pos = positions_all[gidx_arr]
            fwd = forwards_all[gidx_arr]
            ups = ups_all[gidx_arr]
            has_det = np.asarray(
                [self.views[k].has_detection for k in gidxs],
                dtype=bool,
            )

            for li, gi in enumerate(gidxs):
                d = np.linalg.norm(pos - pos[li], axis=1)
                mask = (d > 0.0) & (d <= self.distance_threshold) & has_det
                if self.rotation_threshold_deg is not None:
                    a = batch_relative_rotation_angle_deg(fwd[li], ups[li], fwd, ups)
                    mask = mask & (a <= self.rotation_threshold_deg)
                valid_local = np.where(mask)[0]
                if valid_local.size == 0:
                    continue

                if self.pair_distance_distribution == "natural":
                    if (
                        self.max_pairs_per_image > 0
                        and valid_local.size > self.max_pairs_per_image
                    ):
                        valid_local = rng.choice(
                            valid_local, size=self.max_pairs_per_image, replace=False
                        )
                elif self.pair_distance_distribution == "adaptive":
                    bin_idx = np.clip(
                        np.digitize(d[valid_local], bin_edges) - 1,
                        0,
                        self.n_distance_bins - 1,
                    )
                    bin_groups = [
                        valid_local[bin_idx == b] for b in range(self.n_distance_bins)
                    ]
                    n_per_bin = min(g.size for g in bin_groups)
                    if n_per_bin == 0:
                        continue
                    chunks = []
                    for g in bin_groups:
                        if g.size > n_per_bin:
                            g = rng.choice(g, size=n_per_bin, replace=False)
                        chunks.append(g)
                    valid_local = np.concatenate(chunks)
                else:
                    bin_idx = np.clip(
                        np.digitize(d[valid_local], bin_edges) - 1,
                        0,
                        self.n_distance_bins - 1,
                    )
                    chunks = []
                    for b in range(self.n_distance_bins):
                        in_bin = valid_local[bin_idx == b]
                        if in_bin.size > target_per_bin:
                            in_bin = rng.choice(in_bin, size=target_per_bin, replace=False)
                        chunks.append(in_bin)
                    valid_local = (
                        np.concatenate(chunks) if chunks else valid_local[:0]
                    )
                    if valid_local.size == 0:
                        continue

                # Fallback bin edges for bucket_idx logging when binning is off
                # (natural mode). Always uniform 0..distance_threshold over
                # n_distance_bins so the analytic axis is consistent.
                bucket_edges = bin_edges if bin_edges is not None else np.linspace(
                    0.0, self.distance_threshold, self.n_distance_bins + 1
                )

                view_i = self.views[gi]
                for lj in valid_local.tolist():
                    gj = gidxs[lj]
                    view_j = self.views[gj]
                    action_text = self._build_action_text_for_views(view_i, view_j)
                    target_scores = view_j.scores
                    target_text = scores_to_canonical_json(
                        target_scores,
                        score_keys=self.target_score_keys,
                    )
                    pair_distance = float(d[lj])
                    bucket = int(np.clip(
                        np.digitize([pair_distance], bucket_edges)[0] - 1,
                        0,
                        self.n_distance_bins - 1,
                    ))
                    pair_records.append(
                        PairRecord(
                            index_i=gi,
                            index_j=gj,
                            image_i=view_i.image_path,
                            action_text=action_text,
                            target_text=target_text,
                            target_scores=target_scores,
                            distance=pair_distance,
                            bucket_idx=bucket,
                        )
                    )

        return pair_records

    def _build_zero_action_pairs(self, base_pair_count: int) -> list[PairRecord]:
        if self.zero_action_ratio <= 0.0:
            return []
        detected_indices = [idx for idx, view in enumerate(self.views) if view.has_detection]
        if len(detected_indices) == 0 or base_pair_count == 0:
            return []

        # zero_count / (base_pair_count + zero_count) ~= zero_action_ratio
        zero_count = int(round(base_pair_count * self.zero_action_ratio / (1.0 - self.zero_action_ratio)))
        if zero_count == 0:
            zero_count = 1

        rng = np.random.default_rng(self.seed + 17)
        replace = zero_count > len(detected_indices)
        sampled_indices = rng.choice(np.asarray(detected_indices, dtype=np.int64), size=zero_count, replace=replace)

        zero_pairs: list[PairRecord] = []
        for idx in sampled_indices.tolist():
            view = self.views[idx]
            action_text = self._build_action_text_for_views(view, view)
            target_scores = view.scores
            target_text = scores_to_canonical_json(
                target_scores,
                score_keys=self.target_score_keys,
            )
            zero_pairs.append(
                PairRecord(
                    index_i=idx,
                    index_j=idx,
                    image_i=view.image_path,
                    action_text=action_text,
                    target_text=target_text,
                    target_scores=target_scores,
                )
            )
        return zero_pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, object]:
        pair = self.pairs[idx]
        image = Image.open(pair.image_i).convert("RGB")
        return {
            "image": image,
            "action_text": pair.action_text,
            "target_text": pair.target_text,
            "target_scores": dict(pair.target_scores),
            "index_i": pair.index_i,
            "index_j": pair.index_j,
            "image_path": str(pair.image_i),
            "distance": pair.distance,
            "bucket_idx": pair.bucket_idx,
        }


"""
ex)
- total views: 10,000
- pairs before filtering: 100M 
- nearby pairs: 1.78M
- nearby + detected target: 319K
- zero-action pairs (10% ratio): 35K
- total: 355K
- eval (2%): 7K
"""

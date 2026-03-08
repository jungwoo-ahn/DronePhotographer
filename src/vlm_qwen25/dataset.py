from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .prompt import action_vector_to_text
from .schema import extract_scores, scores_to_canonical_json


@dataclass(frozen=True)
class PairRecord:
    index_i: int
    index_j: int
    image_i: Path
    action_text: str
    target_text: str
    target_scores: dict[str, float]


@dataclass(frozen=True)
class ViewRecord:
    image_path: Path
    camera_position: np.ndarray
    camera_forward: np.ndarray
    camera_up: np.ndarray
    scores: dict[str, float]


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + eps)


def _make_basis(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    fwd = _normalize(forward)
    upn = _normalize(up)
    right = _normalize(np.cross(fwd, upn))
    upn = _normalize(np.cross(right, fwd))
    return np.stack([right, upn, fwd], axis=1)


def _rotation_matrix_to_axis_angle(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = float(np.arccos(cos_theta))
    if theta < 1e-5:
        return np.zeros(3, dtype=np.float32)

    rx = rotation[2, 1] - rotation[1, 2]
    ry = rotation[0, 2] - rotation[2, 0]
    rz = rotation[1, 0] - rotation[0, 1]
    axis = np.array([rx, ry, rz], dtype=np.float32) / (2.0 * np.sin(theta))
    return axis * theta


def relative_rotation_vector(
    forward_i: np.ndarray,
    up_i: np.ndarray,
    forward_j: np.ndarray,
    up_j: np.ndarray,
) -> np.ndarray:
    basis_i = _make_basis(forward_i, up_i)
    basis_j = _make_basis(forward_j, up_j)
    rel = basis_j @ basis_i.T
    return _rotation_matrix_to_axis_angle(rel).astype(np.float32)


class DroneActionScoreDataset(Dataset):
    def __init__(
        self,
        annotations_path: str | Path,
        image_root: str | Path | None = None,
        distance_threshold: float = 1.5,
        max_pairs_per_image: int = 32,
        seed: int = 721,
    ) -> None:
        self.annotations_path = Path(annotations_path)
        self.image_root = Path(image_root) if image_root is not None else self.annotations_path.parent
        self.distance_threshold = float(distance_threshold)
        self.max_pairs_per_image = int(max_pairs_per_image)
        self.seed = int(seed)

        with self.annotations_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self.views = self._load_views(raw)
        self.pairs = self._build_pairs()

    def _load_views(self, raw: list[dict]) -> list[ViewRecord]:
        views: list[ViewRecord] = []
        for item in raw:
            if not item.get("detections"):
                continue

            scores = extract_scores(item)
            image_path = self.image_root / str(item["image"])
            if not image_path.exists():
                raise FileNotFoundError(f"image missing: {image_path}")

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
                    scores=scores,
                )
            )
        return views

    def _build_pairs(self) -> list[PairRecord]:
        rng = np.random.default_rng(self.seed)
        positions = np.stack([v.camera_position for v in self.views], axis=0)

        pair_records: list[PairRecord] = []
        n = len(self.views)
        for i in range(n):
            distances = np.linalg.norm(positions - positions[i], axis=1)
            valid = np.where((distances > 0.0) & (distances <= self.distance_threshold))[0]
            if valid.size == 0:
                continue

            if self.max_pairs_per_image > 0 and valid.size > self.max_pairs_per_image:
                valid = rng.choice(valid, size=self.max_pairs_per_image, replace=False)

            for j in valid.tolist():
                view_i = self.views[i]
                view_j = self.views[j]

                delta_position = tuple((view_j.camera_position - view_i.camera_position).tolist())
                delta_rotation_np = relative_rotation_vector(
                    view_i.camera_forward,
                    view_i.camera_up,
                    view_j.camera_forward,
                    view_j.camera_up,
                )
                delta_rotation = tuple(delta_rotation_np.tolist())

                action_text = action_vector_to_text(delta_position, delta_rotation)
                target_scores = view_j.scores
                target_text = scores_to_canonical_json(target_scores)

                pair_records.append(
                    PairRecord(
                        index_i=i,
                        index_j=j,
                        image_i=view_i.image_path,
                        action_text=action_text,
                        target_text=target_text,
                        target_scores=target_scores,
                    )
                )

        return pair_records

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
        }

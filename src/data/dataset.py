import json
import os
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def _safe_get(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _normalize(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vec / (torch.norm(vec) + eps)


def _make_basis(forward: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    fwd = _normalize(forward)
    upn = _normalize(up)
    right = _normalize(torch.cross(fwd, upn, dim=0))
    upn = _normalize(torch.cross(right, fwd, dim=0))
    return torch.stack([right, upn, fwd], dim=1)


def _rotation_matrix_to_axis_angle(rot: torch.Tensor) -> torch.Tensor:
    trace = torch.trace(rot)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    if theta.item() < 1e-5:
        return torch.zeros(3, dtype=torch.float32)
    rx = rot[2, 1] - rot[1, 2]
    ry = rot[0, 2] - rot[2, 0]
    rz = rot[1, 0] - rot[0, 1]
    axis = torch.stack([rx, ry, rz]).to(torch.float32) / (2.0 * torch.sin(theta))
    return axis * theta


def relative_rotation_vector(
    forward_i: torch.Tensor,
    up_i: torch.Tensor,
    forward_j: torch.Tensor,
    up_j: torch.Tensor,
) -> torch.Tensor:
    basis_i = _make_basis(forward_i, up_i)
    basis_j = _make_basis(forward_j, up_j)
    rel = basis_j @ basis_i.T
    return _rotation_matrix_to_axis_angle(rel)


class DronePairDataset(Dataset):
    def __init__(
        self,
        root: str = "outputs/DogWalk_260204_131211",
        distance_threshold: float = 1.5,
        image_transform: transforms.Compose | None = None,
        pair_cache_path: str | None = None,
    ) -> None:
        self.root = root
        self.distance_threshold = distance_threshold
        self.transform = image_transform or transforms.ToTensor()
        self.pair_cache_path = pair_cache_path

        self.images: List[torch.Tensor] = []
        self.object_names: List[str] = []
        self.object_positions: List[torch.Tensor] = []
        self.camera_positions: List[torch.Tensor] = []
        self.camera_forwards: List[torch.Tensor] = []
        self.camera_ups: List[torch.Tensor] = []
        self.scores: List[Dict[str, Any]] = []

        annotations_path = os.path.join(root, "annotations.json")
        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        total_items = len(data)
        kept_items = 0
        for item in tqdm(data, desc="load annotations"):
            detections = item.get("detections")
            if not detections:
                continue
            kept_items += 1

            image_path = os.path.join(root, item["image"])
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image)

            self.images.append(image_tensor)
            self.object_names.append(item.get("prompt", ""))
            self.object_positions.append(
                torch.tensor(item["object_position"], dtype=torch.float32)
            )
            self.camera_positions.append(
                torch.tensor(item["camera_position"], dtype=torch.float32)
            )
            self.camera_forwards.append(
                torch.tensor(
                    item.get("final_forward", item.get("base_forward")),
                    dtype=torch.float32,
                )
            )
            self.camera_ups.append(
                torch.tensor(
                    item.get("final_up", item.get("base_up")),
                    dtype=torch.float32,
                )
            )

            scores = {
                "score_subject_size_20": item.get("score_subject_size_20"),
                "score_subject_size_50": item.get("score_subject_size_50"),
                "score_subject_size_80": item.get("score_subject_size_80"),
                "score_breathing_space": item.get("score_breathing_space"),
                "score_centeredness": item.get("score_centeredness"),
                "score_rule_of_thirds": item.get("score_rule_of_thirds"),
                "score_rule_of_thirds_line": item.get("score_rule_of_thirds_line"),
                "score_qalign_quality": item.get("score_qalign_quality", item.get("score_qalign")),
                "score_qalign_aesthetic": item.get("score_qalign_aesthetic"),
            }
            self.scores.append(scores)

        self.pairs: List[Tuple[int, int]] = []
        self.delta_pos: torch.Tensor
        self.delta_rot: torch.Tensor
        print(
            f"annotations: total={total_items} kept_after_detection={kept_items}"
        )
        self.stats = {
            "annotations_total": total_items,
            "annotations_kept_after_detection": kept_items,
        }
        self._build_pairs(annotations_path, kept_items)

    def _build_pairs(self, annotations_path: str, kept_items: int) -> None:
        cache_path = self.pair_cache_path
        if cache_path is None:
            cache_path = os.path.join(
                self.root, f"pair_cache_dt{self.distance_threshold:.3f}.pt"
            )
        ann_stat = os.stat(annotations_path)
        cache_meta = {
            "annotations_path": os.path.abspath(annotations_path),
            "annotations_mtime": ann_stat.st_mtime,
            "annotations_size": ann_stat.st_size,
            "distance_threshold": self.distance_threshold,
            "kept_items": kept_items,
        }
        if os.path.exists(cache_path):
            try:
                cached = torch.load(cache_path, map_location="cpu")
                if cached.get("meta") == cache_meta:
                    self.pairs = cached["pairs"]
                    self.delta_pos = cached["delta_pos"]
                    self.delta_rot = cached["delta_rot"]
                    print(
                        f"pairs cache hit: {cache_path} "
                        f"(pairs={len(self.pairs)})"
                    )
                    return
            except Exception:
                pass

        n = len(self.images)
        if n == 0:
            self.delta_pos = torch.empty((0, 3), dtype=torch.float32)
            self.delta_rot = torch.empty((0, 3), dtype=torch.float32)
            self._save_pairs_cache(cache_path, cache_meta)
            return

        cam_pos = torch.stack(self.camera_positions, dim=0)
        delta_pos_list: List[torch.Tensor] = []
        delta_rot_list: List[torch.Tensor] = []

        total_pairs = 0
        kept_pairs = 0
        for i in tqdm(range(n), desc="build pairs"):
            for j in range(n):
                if i == j:
                    continue
                total_pairs += 1
                if torch.norm(cam_pos[i] - cam_pos[j]).item() > self.distance_threshold:
                    continue
                self.pairs.append((i, j))
                kept_pairs += 1
                delta_pos_list.append(self.camera_positions[j] - self.camera_positions[i])
                delta_rot_list.append(
                    relative_rotation_vector(
                        self.camera_forwards[i],
                        self.camera_ups[i],
                        self.camera_forwards[j],
                        self.camera_ups[j],
                    )
                )

        if delta_pos_list:
            self.delta_pos = torch.stack(delta_pos_list, dim=0)
            self.delta_rot = torch.stack(delta_rot_list, dim=0)
        else:
            self.delta_pos = torch.empty((0, 3), dtype=torch.float32)
            self.delta_rot = torch.empty((0, 3), dtype=torch.float32)
        ratio = 0.0 if total_pairs == 0 else (kept_pairs / total_pairs)
        print(
            f"pairs: total={total_pairs} kept_after_distance={kept_pairs} "
            f"ratio={ratio:.4f} distance_threshold={self.distance_threshold}"
        )
        self.stats.update(
            {
                "pairs_total": total_pairs,
                "pairs_kept_after_distance": kept_pairs,
                "pairs_kept_ratio": ratio,
                "distance_threshold": self.distance_threshold,
            }
        )
        self._save_pairs_cache(cache_path, cache_meta)

    def _save_pairs_cache(self, cache_path: str, cache_meta: Dict[str, Any]) -> None:
        try:
            torch.save(
                {
                    "meta": cache_meta,
                    "pairs": self.pairs,
                    "delta_pos": self.delta_pos,
                    "delta_rot": self.delta_rot,
                },
                cache_path,
            )
            print(f"pairs cache saved: {cache_path}")
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        i, j = self.pairs[idx]
        image = self.images[i]
        delta_pos = self.delta_pos[idx]
        delta_rot = self.delta_rot[idx]
        score_j = self.scores[j]
        targets = torch.tensor(
            [
                float(score_j["score_rule_of_thirds_line"]),
                # float(score_j["score_qalign_quality"]),
                # float(score_j["score_qalign_aesthetic"]),
                float(score_j["score_breathing_space"]),
                float(score_j["score_centeredness"]),
                # float(score_j["score_rule_of_thirds"]),
                float(score_j["score_subject_size_20"]),
                # float(score_j["score_subject_size_50"]),
                float(score_j["score_subject_size_80"]),
            ],
            dtype=torch.float32,
        )
        return image, delta_pos, delta_rot, targets

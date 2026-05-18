from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Allow importing scripts/generate_target.py as a module.
REPO_ROOT = Path(__file__).resolve().parents[1]
_scripts_dir = str(REPO_ROOT / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


@pytest.fixture
def valid_full_target() -> dict[str, float]:
    """Complete target with composition + c2o keys, orthogonal unit vectors."""
    return {
        "center_x": 0.5,
        "center_y": 0.5,
        "occupancy": 0.4,
        "aspect_ratio": 1.0,
        "camera_to_object_fx": 0.0,
        "camera_to_object_fy": 1.0,
        "camera_to_object_fz": 0.0,
        "camera_to_object_ux": 0.0,
        "camera_to_object_uy": 0.0,
        "camera_to_object_uz": 1.0,
    }


@pytest.fixture
def minimal_valid_target() -> dict[str, float]:
    """Minimal target: occupancy + 6 c2o keys."""
    return {
        "occupancy": 0.4,
        "camera_to_object_fx": 0.0,
        "camera_to_object_fy": 1.0,
        "camera_to_object_fz": 0.0,
        "camera_to_object_ux": 0.0,
        "camera_to_object_uy": 0.0,
        "camera_to_object_uz": 1.0,
    }


def dot3(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm3(v: tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in v))

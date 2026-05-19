"""Asset naming helpers (pure Python, no bpy dependency)."""

from __future__ import annotations

from pathlib import Path


def scene_name(path: Path) -> str:
    """Derive a human-readable scene name from a .blend file path."""
    if path.parent.name not in ("scenes", "assets"):
        return path.parent.name
    return path.stem


def object_name(path: Path) -> str:
    """Derive a human-readable object name from an asset file path."""
    if path.parent.name not in ("objects", "assets"):
        return path.parent.name
    return path.stem

"""Asset discovery utilities (pure Python, no bpy dependency)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

SCENE_EXTS = {".blend"}
OBJECT_EXTS = {".blend", ".obj", ".glb", ".gltf", ".fbx", ".ply"}
HDRI_EXTS = {".hdr", ".exr"}
PICK_EXTS = [".glb", ".gltf", ".fbx", ".obj", ".blend"]


def discover_files(directory: str | Path, exts: set[str]) -> list[Path]:
    """Recursively find files matching any of the given extensions.

    Follows symlinks so that directories containing symlinked subdirectories
    (e.g. data/objects/ with symlinks to asset folders) are traversed correctly.
    """
    import os

    d = Path(directory)
    if not d.is_dir():
        return []
    ext_set = {e.lower() for e in exts}
    found = []
    for root, _dirs, files in os.walk(d, followlinks=True):
        for fname in files:
            if Path(fname).suffix.lower() in ext_set:
                found.append(Path(root) / fname)
    return sorted(set(found))


def find_scene_blend(scene_path: Path) -> Path | None:
    """Given a path (file or directory), return the first .blend file found."""
    if scene_path.suffix == ".blend" and scene_path.is_file():
        return scene_path
    if scene_path.is_dir():
        blends = sorted(scene_path.rglob("*.blend"))
        return blends[0] if blends else None
    return None


def pick_first_file(path: Path, exts: list[str] | None = None) -> Path | None:
    """Return the first file matching priority-ordered extensions."""
    if path.is_file():
        return path
    if not path.exists():
        return None
    for ext in (exts or PICK_EXTS):
        for f in sorted(path.rglob(f"*{ext}")):
            return f
    return None


def filter_scenes_by_texture(
    scene_files: list[Path],
    texture_check_path: str | None,
    quality_filter: list[str],
) -> list[Path]:
    """Filter scene files by texture quality from a texture_check JSON."""
    if not texture_check_path or not Path(texture_check_path).exists():
        log.warning("No texture_check.json found, using all scenes")
        return scene_files

    with open(texture_check_path) as f:
        tex_data = json.load(f)

    usable_blends = set()
    for entry in tex_data:
        if entry.get("status") in quality_filter:
            usable_blends.add(entry.get("blend"))
            usable_blends.add(entry.get("name"))

    filtered = []
    for sf in scene_files:
        name = sf.parent.name if sf.parent.name != "scenes" else sf.stem
        if str(sf) in usable_blends or sf.stem in usable_blends or name in usable_blends:
            filtered.append(sf)

    log.info(f"Texture filter: {len(filtered)}/{len(scene_files)} scenes usable")
    return filtered

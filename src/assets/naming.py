"""Asset naming helpers (pure Python, no bpy dependency)."""

from __future__ import annotations

import re
from pathlib import Path

# What Windows forbids inside a path component, plus the separators and control
# characters. An asset name is allowed to contain any of these -- "Retro-Chic: Urban
# Stroll" is simply what the asset is called -- but a *filename* built from one is not,
# and a repository holding such a filename cannot be checked out on Windows at all.
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}


def path_safe(name: str) -> str:
    """Turn a name into one path component that every OS can store.

    Use this where a name becomes a file or directory name, and *not* where it is the
    asset's identity: the identity has to keep matching the real directory under
    `data/objects/`, which may well have a colon in it.
    """
    out = _UNSAFE.sub("-", name).rstrip(". ")
    if out.split(".")[0].upper() in _RESERVED:
        out = "_" + out
    return out or "_"


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

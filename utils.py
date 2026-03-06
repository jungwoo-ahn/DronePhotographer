from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as f:
        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except Exception as e:
                raise ImportError("PyYAML is required for YAML configs. Install `pyyaml`.") from e
            data = yaml.safe_load(f) or {}
        else:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


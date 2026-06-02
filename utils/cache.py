from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def cache_key(source_path: str | Path, config: Mapping[str, Any]) -> str:
    payload = {
        "source_path": str(Path(source_path)),
        "config": config,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.pkl"


def is_cache_hit(path: str | Path) -> bool:
    return Path(path).exists()

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping


def cache_key(source_path: str | Path, config: Mapping[str, Any]) -> str:
    path = Path(source_path)
    stat = path.stat() if path.exists() else None
    payload = {
        "source_path": str(path.resolve()),
        "source_size": stat.st_size if stat else None,
        "source_mtime_ns": stat.st_mtime_ns if stat else None,
        "config": config,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.pkl"


def is_cache_hit(path: str | Path) -> bool:
    return Path(path).exists()


def load_cache(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_cache(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

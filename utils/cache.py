from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np


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
        return _from_cache_payload(pickle.load(f))


def save_cache(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(_to_cache_payload(value), f, protocol=pickle.HIGHEST_PROTOCOL)


def _to_cache_payload(value: Any) -> Any:
    if _is_open3d_point_cloud(value):
        return {
            "__cache_type__": "open3d_point_cloud",
            "points": np.asarray(value.points),
            "normals": np.asarray(value.normals),
            "colors": np.asarray(value.colors),
        }
    if _is_open3d_feature(value):
        return {
            "__cache_type__": "open3d_feature",
            "data": np.asarray(value.data),
        }
    if isinstance(value, dict):
        return {k: _to_cache_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cache_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cache_payload(v) for v in value)
    return value


def _from_cache_payload(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__cache_type__") == "open3d_point_cloud":
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        points = np.asarray(value.get("points", []), dtype=float)
        normals = np.asarray(value.get("normals", []), dtype=float)
        colors = np.asarray(value.get("colors", []), dtype=float)
        pcd.points = o3d.utility.Vector3dVector(points)
        if len(normals) == len(points):
            pcd.normals = o3d.utility.Vector3dVector(normals)
        if len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    if isinstance(value, dict) and value.get("__cache_type__") == "open3d_feature":
        import open3d as o3d

        feature = o3d.pipelines.registration.Feature()
        feature.data = np.asarray(value.get("data", []), dtype=float)
        return feature
    if isinstance(value, dict):
        return {k: _from_cache_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_cache_payload(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_from_cache_payload(v) for v in value)
    return value


def _is_open3d_point_cloud(value: Any) -> bool:
    return value.__class__.__name__ == "PointCloud" and "open3d" in value.__class__.__module__


def _is_open3d_feature(value: Any) -> bool:
    return value.__class__.__name__ == "Feature" and "open3d" in value.__class__.__module__

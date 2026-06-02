from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def read_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        if path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f) or {}


def read_point_cloud(path: str | Path) -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required to read point clouds.") from exc
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Point cloud is empty or unreadable: {path}")
    return pcd


def write_point_cloud(path: str | Path, pcd: Any) -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required to write point clouds.") from exc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), pcd):
        raise RuntimeError(f"Failed to write point cloud: {path}")


def read_matrix(path: str | Path) -> np.ndarray:
    matrix = np.loadtxt(Path(path), dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Matrix must be 4x4: {path}")
    return matrix


def write_matrix(path: str | Path, matrix: Any) -> None:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError("Matrix must be 4x4.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.10f")

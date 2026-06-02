from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from utils import io


def transformed_cloud(source_pcd: Any, transformation: Any) -> Any:
    source = _clone_point_cloud(source_pcd)
    source.transform(np.asarray(transformation, dtype=float))
    return source


def save_transformed_cloud(source_pcd: Any, transformation: Any, path: str | Path) -> None:
    io.write_point_cloud(path, transformed_cloud(source_pcd, transformation))


def save_overlay_cloud(source_pcd: Any, target_pcd: Any, transformation: Any, path: str | Path) -> None:
    transformed = transformed_cloud(source_pcd, transformation)
    target = _clone_point_cloud(target_pcd)
    target.paint_uniform_color([0.55, 0.55, 0.55])
    transformed.paint_uniform_color([1.0, 0.85, 0.05])
    io.write_point_cloud(path, target + transformed)


def draw_overlay(source_pcd: Any, target_pcd: Any, transformation: Any) -> None:
    import open3d as o3d

    transformed = transformed_cloud(source_pcd, transformation)
    target = _clone_point_cloud(target_pcd)
    target.paint_uniform_color([0.55, 0.55, 0.55])
    transformed.paint_uniform_color([1.0, 0.85, 0.05])
    o3d.visualization.draw_geometries([target, transformed])


def _clone_point_cloud(pcd: Any) -> Any:
    import copy

    return pcd.clone() if hasattr(pcd, "clone") else copy.deepcopy(pcd)

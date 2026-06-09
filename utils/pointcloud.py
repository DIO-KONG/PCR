from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import open3d as o3d


def read_point_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    path = Path(path)
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Point cloud is empty or unreadable: {path}")
    return cloud


def write_point_cloud(path: str | Path, cloud: o3d.geometry.PointCloud) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f"Open3D failed to write point cloud: {path}")


def clone_transform(cloud: o3d.geometry.PointCloud, matrix: np.ndarray) -> o3d.geometry.PointCloud:
    transformed = copy.deepcopy(cloud)
    transformed.transform(matrix)
    return transformed


def make_overlay(
    target: o3d.geometry.PointCloud,
    transformed_source: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    target_copy = copy.deepcopy(target)
    source_copy = copy.deepcopy(transformed_source)
    target_copy.paint_uniform_color([0.55, 0.55, 0.55])
    source_copy.paint_uniform_color([1.0, 0.85, 0.05])
    return target_copy + source_copy


def point_count(cloud: o3d.geometry.PointCloud) -> int:
    return int(np.asarray(cloud.points).shape[0])

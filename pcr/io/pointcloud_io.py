from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def load_point_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    """读取非空点云。"""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file does not exist: {path}")
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Point cloud is empty or unreadable: {path}")
    return cloud


def save_point_cloud(path: str | Path, cloud: o3d.geometry.PointCloud) -> None:
    """保存非空点云。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cloud.is_empty():
        raise ValueError(f"Refuse to write an empty point cloud: {path}")
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f"Open3D failed to write point cloud: {path}")


def clone_point_cloud(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """复制点云，避免算法原地修改输入。"""

    return o3d.geometry.PointCloud(cloud)


def make_registration_overlay(
    target_world: o3d.geometry.PointCloud,
    transformed_source: o3d.geometry.PointCloud,
) -> o3d.geometry.PointCloud:
    """生成灰色 target/world + 黄色 source 的检查点云。"""

    target_copy = clone_point_cloud(target_world)
    source_copy = clone_point_cloud(transformed_source)
    target_copy.paint_uniform_color((0.55, 0.55, 0.55))
    source_copy.paint_uniform_color((1.0, 0.82, 0.05))
    return target_copy + source_copy


def point_cloud_from_points(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    """把 Nx3 数组转换成统一颜色点云，用于轻量调试 artifact。"""

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=float))
    cloud.paint_uniform_color(color)
    return cloud


def make_shared_frame_overlay(
    *,
    target_points: np.ndarray,
    transformed_source_points: np.ndarray,
) -> o3d.geometry.PointCloud:
    """生成共享帧对应点 overlay。

    坐标系约定为 target batch 坐标：target/baseline 点为灰色，source/window
    点先用 coarse source->target 变换后染成黄色。这个 artifact 专门用于检查
    shared-frame 粗配准，而不是检查 global world 融合。
    """

    target_cloud = point_cloud_from_points(target_points, (0.55, 0.55, 0.55))
    source_cloud = point_cloud_from_points(transformed_source_points, (1.0, 0.82, 0.05))
    return target_cloud + source_cloud

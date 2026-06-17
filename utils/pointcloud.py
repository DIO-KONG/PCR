from __future__ import annotations

import copy
from pathlib import Path

import open3d as o3d


def clone_cloud(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """复制点云，避免算法原地修改输入。"""

    return copy.deepcopy(cloud)


def load_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    """读取点云并检查空点云。"""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file does not exist: {path}")
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Point cloud is empty or unreadable: {path}")
    return cloud


def save_cloud(path: str | Path, cloud: o3d.geometry.PointCloud) -> None:
    """保存点云，自动创建父目录。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cloud.is_empty():
        raise ValueError(f"Refuse to write an empty point cloud: {path}")
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f"Open3D failed to write point cloud: {path}")


def transform_cloud(cloud: o3d.geometry.PointCloud, matrix) -> o3d.geometry.PointCloud:
    """复制点云并应用 4x4 变换。"""

    result = clone_cloud(cloud)
    result.transform(matrix)
    return result


def make_overlay(
    target: o3d.geometry.PointCloud,
    transformed_source: o3d.geometry.PointCloud,
    *,
    target_color: tuple[float, float, float] = (0.55, 0.55, 0.55),
    source_color: tuple[float, float, float] = (1.0, 0.82, 0.05),
) -> o3d.geometry.PointCloud:
    """生成配准检查 overlay。

    约定 target/world 为灰色，变换后的 source 为黄色，便于快速看重合关系。
    """

    target_copy = clone_cloud(target)
    source_copy = clone_cloud(transformed_source)
    target_copy.paint_uniform_color(target_color)
    source_copy.paint_uniform_color(source_color)
    return target_copy + source_copy

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class FusionResult:
    """一次融合的完整结果。

    说明：
    - `cloud` 是最终融合后的地图，会继续作为下一步 walk-forward 的 target。
    - `stats` 记录本次融合接受、丢弃、判定为重复或冲突的点数，方便后续调参。
    - `debug_clouds` 用于保存可视化辅助点云，例如“真正加入的新点”和“被判定为重影冲突的点”。
    """

    cloud: o3d.geometry.PointCloud
    stats: dict[str, int | float | str]
    debug_clouds: dict[str, o3d.geometry.PointCloud]


def _select_points(
    cloud: o3d.geometry.PointCloud,
    mask: np.ndarray,
) -> o3d.geometry.PointCloud:
    """按照布尔掩码从点云中取出子点云，并保留颜色。

    Open3D 自带 `select_by_index` 也能完成类似工作，但这里直接用 numpy 构造，
    是为了让点、颜色、法线等字段的处理逻辑更清晰，后续扩展权重或观测次数也更容易。
    """

    selected = o3d.geometry.PointCloud()
    points = np.asarray(cloud.points)
    selected.points = o3d.utility.Vector3dVector(points[mask])

    colors = np.asarray(cloud.colors)
    if colors.shape[0] == points.shape[0]:
        selected.colors = o3d.utility.Vector3dVector(colors[mask])
    return selected


def _nearest_distances(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> np.ndarray:
    """计算 source 每个点到 target 的最近邻距离。

    这里用于判断“source 点是否已经被地图解释过”。距离很小视作重复观测；
    距离略大但仍靠近已有表面时，通常是配准误差或深度漂移造成的重影候选；
    距离明显较大时，才认为它可能代表新增区域。
    """

    source_points = np.asarray(source.points)
    if source_points.size == 0:
        return np.asarray([], dtype=float)

    target_points = np.asarray(target.points)
    if target_points.size == 0:
        return np.full(len(source_points), np.inf, dtype=float)

    tree = o3d.geometry.KDTreeFlann(target)
    distances = np.empty(len(source_points), dtype=float)
    for index, point in enumerate(source_points):
        count, _, sq_distances = tree.search_knn_vector_3d(point, 1)
        distances[index] = float(np.sqrt(sq_distances[0])) if count else np.inf
    return distances


def _postprocess_merged_cloud(
    merged: o3d.geometry.PointCloud,
    *,
    fusion_voxel_size: float,
    sor_enabled: bool,
    sor_nb_neighbors: int,
    sor_std_ratio: float,
) -> o3d.geometry.PointCloud:
    """对融合后的地图做统一后处理。

    后处理保持轻量：先做体素降采样控制密度，再按需做统计离群点移除。
    这一步不负责判断重影，重影判断在融合策略内部完成。
    """

    if fusion_voxel_size > 0:
        merged = merged.voxel_down_sample(fusion_voxel_size)
    if sor_enabled:
        merged, _ = merged.remove_statistical_outlier(
            nb_neighbors=sor_nb_neighbors,
            std_ratio=sor_std_ratio,
        )
    return merged


def fuse_world_with_stats(
    target_world: o3d.geometry.PointCloud,
    incremental_source: o3d.geometry.PointCloud,
    matrix,
    *,
    fusion_voxel_size: float,
    method: str = "simple",
    duplicate_distance: float = 0.08,
    conflict_distance: float = 0.20,
    sor_enabled: bool = False,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
) -> FusionResult:
    """融合 source 到 target，并返回统计信息。

    当前支持两种策略：
    - `simple`：历史行为，直接把配准后的 source 追加到 target，再体素降采样。
    - `conflict_filter`：当前实现的重影检测/冲突过滤策略。

    `conflict_filter` 的核心规则：
    - source 点到 target 最近邻距离 <= `duplicate_distance`：认为地图中已有该表面，丢弃以保持密度不变。
    - 距离在 (`duplicate_distance`, `conflict_distance`]：认为靠近已有表面但没有对齐好，作为重影冲突丢弃。
    - 距离 > `conflict_distance`：认为可能是新增区域，允许加入地图。
    """

    transformed = copy.deepcopy(incremental_source)
    transformed.transform(matrix)
    source_points = len(transformed.points)
    target_points = len(target_world.points)

    if method == "simple":
        merged = target_world + transformed
        cloud = _postprocess_merged_cloud(
            merged,
            fusion_voxel_size=fusion_voxel_size,
            sor_enabled=sor_enabled,
            sor_nb_neighbors=sor_nb_neighbors,
            sor_std_ratio=sor_std_ratio,
        )
        return FusionResult(
            cloud=cloud,
            stats={
                "fusion_method": method,
                "target_points_before": target_points,
                "source_points_after_transform": source_points,
                "accepted_new_points": source_points,
                "duplicate_points": 0,
                "conflict_points": 0,
                "fused_points_after": len(cloud.points),
            },
            debug_clouds={"transformed_source": transformed},
        )

    if method != "conflict_filter":
        raise ValueError(f"Unknown fusion method: {method}")
    if duplicate_distance < 0 or conflict_distance < duplicate_distance:
        raise ValueError("conflict_distance must be >= duplicate_distance >= 0")

    distances = _nearest_distances(transformed, target_world)
    duplicate_mask = distances <= duplicate_distance
    conflict_mask = (distances > duplicate_distance) & (distances <= conflict_distance)
    accepted_mask = distances > conflict_distance

    accepted_cloud = _select_points(transformed, accepted_mask)
    conflict_cloud = _select_points(transformed, conflict_mask)
    duplicate_cloud = _select_points(transformed, duplicate_mask)

    merged = target_world + accepted_cloud
    cloud = _postprocess_merged_cloud(
        merged,
        fusion_voxel_size=fusion_voxel_size,
        sor_enabled=sor_enabled,
        sor_nb_neighbors=sor_nb_neighbors,
        sor_std_ratio=sor_std_ratio,
    )

    return FusionResult(
        cloud=cloud,
        stats={
            "fusion_method": method,
            "target_points_before": target_points,
            "source_points_after_transform": source_points,
            "accepted_new_points": int(np.count_nonzero(accepted_mask)),
            "duplicate_points": int(np.count_nonzero(duplicate_mask)),
            "conflict_points": int(np.count_nonzero(conflict_mask)),
            "duplicate_distance": float(duplicate_distance),
            "conflict_distance": float(conflict_distance),
            "fused_points_after": len(cloud.points),
        },
        debug_clouds={
            "accepted_new_points": accepted_cloud,
            "rejected_conflict_points": conflict_cloud,
            "duplicate_points": duplicate_cloud,
            "transformed_source": transformed,
        },
    )


def fuse_world(
    target_world: o3d.geometry.PointCloud,
    incremental_source: o3d.geometry.PointCloud,
    matrix,
    *,
    fusion_voxel_size: float,
    sor_enabled: bool = False,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """兼容旧调用的简单融合接口。

    新实验建议使用 `fuse_world_with_stats`，这样可以获得冲突过滤统计和调试点云。
    """

    return fuse_world_with_stats(
        target_world,
        incremental_source,
        matrix,
        fusion_voxel_size=fusion_voxel_size,
        method="simple",
        sor_enabled=sor_enabled,
        sor_nb_neighbors=sor_nb_neighbors,
        sor_std_ratio=sor_std_ratio,
    ).cloud

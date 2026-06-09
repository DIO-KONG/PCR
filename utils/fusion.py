from __future__ import annotations

import copy

import open3d as o3d


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
    transformed = copy.deepcopy(incremental_source)
    transformed.transform(matrix)
    merged = target_world + transformed
    if fusion_voxel_size > 0:
        merged = merged.voxel_down_sample(fusion_voxel_size)
    if sor_enabled:
        merged, _ = merged.remove_statistical_outlier(
            nb_neighbors=sor_nb_neighbors,
            std_ratio=sor_std_ratio,
        )
    return merged

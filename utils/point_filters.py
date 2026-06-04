from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def subset_point_cloud(pcd: Any, subset: str, config: Mapping[str, Any] | None = None) -> Any:
    config = dict(config or {})
    subset = str(subset or "all").lower()
    if subset == "all":
        return _clone_point_cloud(pcd)

    points = np.asarray(pcd.points)
    if len(points) == 0:
        return _clone_point_cloud(pcd)

    if subset == "near_mid_only":
        radius = np.linalg.norm(points[:, [0, 2]], axis=1)
        max_radius = float(config.get("near_mid_radius", 10.0))
        mask = radius <= max_radius
    elif subset == "non_floor":
        # Robot frame: -Y is height, so floor/low structures are near the maximum Y.
        floor_y = float(config.get("floor_y", np.max(points[:, 1])))
        band = float(config.get("floor_band", 0.35))
        mask = points[:, 1] < floor_y - band
    elif subset == "high_curvature":
        mask = _high_curvature_mask(pcd, config)
    else:
        raise ValueError(f"Unsupported point subset: {subset}")

    idx = np.flatnonzero(mask)
    if len(idx) < int(config.get("min_subset_points", 50)):
        return _clone_point_cloud(pcd)
    return pcd.select_by_index(idx.tolist())


def prepare_subset_features(o3d: Any, pcd: Any, voxel_size: float, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or {})
    pcd_down = pcd.voxel_down_sample(float(voxel_size))
    normal_radius = float(voxel_size) * float(config.get("normal_radius_factor", 2.0))
    fpfh_radius = float(voxel_size) * float(config.get("fpfh_radius_factor", 5.0))
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=int(config.get("normal_max_nn", 30)),
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_radius,
            max_nn=int(config.get("fpfh_max_nn", 100)),
        ),
    )
    return {"pcd_down": pcd_down, "fpfh": fpfh, "voxel_size": float(voxel_size)}


def _high_curvature_mask(pcd: Any, config: Mapping[str, Any]) -> np.ndarray:
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    k = int(config.get("curvature_knn", 20))
    quantile = float(config.get("curvature_quantile", 0.75))

    import open3d as o3d

    tree = o3d.geometry.KDTreeFlann(pcd)
    curvature = np.zeros(len(points), dtype=float)
    for i, point in enumerate(points):
        count, idx, _ = tree.search_knn_vector_3d(point, k)
        if count < 4:
            continue
        neighbors = points[np.asarray(idx, dtype=int)]
        cov = np.cov((neighbors - neighbors.mean(axis=0)).T)
        eig = np.sort(np.linalg.eigvalsh(cov))
        denom = float(np.sum(eig))
        curvature[i] = float(eig[0] / denom) if denom > 0.0 else 0.0
    threshold = float(np.quantile(curvature, quantile))
    return curvature >= threshold


def _clone_point_cloud(pcd: Any) -> Any:
    import copy

    return pcd.clone() if hasattr(pcd, "clone") else copy.deepcopy(pcd)

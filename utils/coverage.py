from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def inlier_pairs(source_pcd: Any, target_pcd: Any, transform: np.ndarray, max_distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import open3d as o3d

    source_points = np.asarray(source_pcd.points, dtype=float)
    target_points = np.asarray(target_pcd.points, dtype=float)
    transformed = (transform[:3, :3] @ source_points.T).T + transform[:3, 3]
    tree = o3d.geometry.KDTreeFlann(target_pcd)
    src = []
    tgt = []
    distances = []
    for source_point, transformed_point in zip(source_points, transformed):
        count, idx, squared = tree.search_knn_vector_3d(transformed_point, 1)
        if count:
            distance = float(np.sqrt(squared[0]))
            if distance <= max_distance:
                src.append(source_point)
                tgt.append(target_points[idx[0]])
                distances.append(distance)
    return np.asarray(src, dtype=float), np.asarray(tgt, dtype=float), np.asarray(distances, dtype=float)


def coverage_score(points: np.ndarray, reference_points: np.ndarray, config: Mapping[str, Any] | None = None) -> float | None:
    config = dict(config or {})
    if len(points) == 0 or len(reference_points) == 0:
        return None
    grid_size = float(config.get("coverage_grid_size", 1.0))
    min_bound = np.min(reference_points, axis=0)
    max_bound = np.max(reference_points, axis=0)
    extent = np.maximum(max_bound - min_bound, grid_size)
    dims = np.maximum(np.ceil(extent / grid_size).astype(int), 1)
    dims = np.minimum(dims, int(config.get("coverage_max_bins_per_axis", 64)))
    normalized = (points - min_bound) / extent
    idx = np.floor(normalized * dims).astype(int)
    valid = np.all((idx >= 0) & (idx < dims), axis=1)
    if not np.any(valid):
        return 0.0
    occupied = {tuple(row) for row in idx[valid]}
    total = int(np.prod(dims))
    return float(len(occupied) / max(total, 1))

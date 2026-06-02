from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from utils.transforms import orthogonality_error


def evaluate_registration(source_pcd: Any, target_pcd: Any, transformation: Any, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or {})
    matrix = np.asarray(transformation, dtype=float)
    record: dict[str, Any] = {
        "orthogonality_error": orthogonality_error(matrix),
        "determinant": float(np.linalg.det(matrix[:3, :3])),
    }

    try:
        distances = _nearest_neighbor_distances(source_pcd, target_pcd, matrix)
        if distances.size:
            record["rmse"] = float(np.sqrt(np.mean(distances**2)))
            trimmed_ratio = float(config.get("trimmed_ratio", 0.9))
            keep = max(1, int(len(distances) * min(max(trimmed_ratio, 0.0), 1.0)))
            record["trimmed_distance"] = float(np.mean(np.sort(distances)[:keep]))
    except Exception as exc:
        record["metric_error"] = str(exc)

    return record


def _nearest_neighbor_distances(source_pcd: Any, target_pcd: Any, transformation: np.ndarray) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for geometric metrics.") from exc

    source = _clone_point_cloud(source_pcd)
    source.transform(transformation.copy())
    tree = o3d.geometry.KDTreeFlann(target_pcd)
    distances = []
    for point in np.asarray(source.points):
        count, _, squared = tree.search_knn_vector_3d(point, 1)
        if count:
            distances.append(float(np.sqrt(squared[0])))
    return np.asarray(distances, dtype=float)


def _clone_point_cloud(pcd: Any) -> Any:
    import copy

    return pcd.clone() if hasattr(pcd, "clone") else copy.deepcopy(pcd)

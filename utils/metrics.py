from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from utils.transforms import orthogonality_error


METRIC_KEYS = (
    "fitness",
    "inlier_rmse",
    "median_nn_dist",
    "trimmed_mean_nn_dist",
    "overlap_ratio",
    "det_R",
    "orthogonality_error",
    "translation_norm",
)


def empty_registration_metrics(error: str | None = None) -> dict[str, Any]:
    record = {key: None for key in METRIC_KEYS}
    if error:
        record["metric_error"] = error
    return record


def evaluate_registration(
    source_pcd: Any,
    target_pcd: Any,
    transformation: Any,
    config: Mapping[str, Any] | None = None,
    registration_result: Any | None = None,
) -> dict[str, Any]:
    config = dict(config or {})
    matrix = np.asarray(transformation, dtype=float)
    record: dict[str, Any] = empty_registration_metrics()
    record.update(
        {
            "det_R": float(np.linalg.det(matrix[:3, :3])),
            "orthogonality_error": orthogonality_error(matrix),
            "translation_norm": float(np.linalg.norm(matrix[:3, 3])),
        }
    )

    result_fitness = getattr(registration_result, "fitness", None)
    result_rmse = getattr(registration_result, "inlier_rmse", None)
    record.update(
        {
            "fitness": float(result_fitness) if result_fitness is not None else None,
            "inlier_rmse": float(result_rmse) if result_rmse is not None else None,
        }
    )

    try:
        distances = _nearest_neighbor_distances(source_pcd, target_pcd, matrix)
        if distances.size:
            trimmed_ratio = float(config.get("trimmed_ratio", 0.9))
            trimmed_ratio = min(max(trimmed_ratio, 0.0), 1.0)
            keep = max(1, int(len(distances) * trimmed_ratio))
            overlap_threshold = float(config.get("overlap_threshold", config.get("max_correspondence_distance", 0.1)))
            inlier_distances = distances[distances <= overlap_threshold]
            record.update(
                {
                    "median_nn_dist": float(np.median(distances)),
                    "trimmed_mean_nn_dist": float(np.mean(np.sort(distances)[:keep])),
                    "overlap_ratio": float(len(inlier_distances) / len(distances)),
                }
            )
            if record["fitness"] is None:
                record["fitness"] = record["overlap_ratio"]
            if record["inlier_rmse"] is None and len(inlier_distances):
                record["inlier_rmse"] = float(np.sqrt(np.mean(inlier_distances**2)))
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

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from utils.transforms import orthogonality_error


METRIC_KEYS = (
    "eval_fitness",
    "eval_inlier_rmse",
    "eval_median_nn_dist",
    "eval_trimmed_mean_nn_dist",
    "eval_overlap_ratio",
    "eval_det_R",
    "eval_orthogonality_error",
    "eval_translation_norm",
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
            "eval_det_R": float(np.linalg.det(matrix[:3, :3])),
            "eval_orthogonality_error": orthogonality_error(matrix),
            "eval_translation_norm": float(np.linalg.norm(matrix[:3, 3])),
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
            eval_fitness = float(len(inlier_distances) / len(distances))
            record.update(
                {
                    "eval_fitness": eval_fitness,
                    "eval_median_nn_dist": float(np.median(distances)),
                    "eval_trimmed_mean_nn_dist": float(np.mean(np.sort(distances)[:keep])),
                    "eval_overlap_ratio": eval_fitness,
                }
            )
            if len(inlier_distances):
                record["eval_inlier_rmse"] = float(np.sqrt(np.mean(inlier_distances**2)))
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

from __future__ import annotations

import numpy as np
import open3d as o3d


def nearest_neighbor_distances(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> np.ndarray:
    """计算 source 到 target 的最近邻距离数组。"""

    target_tree = o3d.geometry.KDTreeFlann(target)
    target_points = np.asarray(target.points)
    source_points = np.asarray(source.points)
    if source_points.size == 0 or target_points.size == 0:
        return np.asarray([], dtype=float)

    distances = np.empty(len(source_points), dtype=float)
    for index, point in enumerate(source_points):
        count, _, sq_distances = target_tree.search_knn_vector_3d(point, 1)
        distances[index] = float(np.sqrt(sq_distances[0])) if count else np.inf
    return distances


def evaluate_registration(
    transformed_source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    overlap_threshold: float,
    trimmed_ratio: float,
) -> dict[str, float | None]:
    """评估配准质量。

    fitness 表示多少 source 点落在 overlap 阈值内；
    trimmed mean 用最近的若干比例点计算平均距离，比全量均值更不容易被少量远点拉坏。
    """

    distances = nearest_neighbor_distances(transformed_source, target)
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return {
            "eval_fitness": 0.0,
            "eval_inlier_rmse": None,
            "eval_median_nn_dist": None,
            "eval_trimmed_mean_nn_dist": None,
        }

    inliers = finite[finite <= overlap_threshold]
    trimmed_count = max(1, int(len(finite) * trimmed_ratio))
    trimmed = np.sort(finite)[:trimmed_count]
    return {
        "eval_fitness": float(len(inliers) / len(finite)),
        "eval_inlier_rmse": float(np.sqrt(np.mean(inliers * inliers))) if len(inliers) else None,
        "eval_median_nn_dist": float(np.median(finite)),
        "eval_trimmed_mean_nn_dist": float(np.mean(trimmed)),
    }


def matrix_diagnostics(matrix: np.ndarray) -> dict[str, float]:
    """给 4x4 变换矩阵做诊断。

    因为当前算法允许尺度，`det` 和奇异值不一定等于 1。
    这些指标用于判断尺度是否异常，而不是判定矩阵是否合法。
    """

    linear = matrix[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    gram = linear.T @ linear
    scale_sq = float(np.mean(np.diag(gram)))
    orthogonality_error = float(np.linalg.norm(gram / max(scale_sq, 1e-12) - np.eye(3)))
    return {
        "eval_det_R": float(np.linalg.det(linear)),
        "eval_orthogonality_error": orthogonality_error,
        "eval_translation_norm": float(np.linalg.norm(matrix[:3, 3])),
        "scale_min": float(np.min(singular_values)),
        "scale_max": float(np.max(singular_values)),
    }


def quality_status(metrics: dict, gate: dict) -> str:
    """根据数值门槛给结果打状态标签。

    该状态只表示“数值上是否可疑”，不能替代 overlay 视觉检查。
    """

    fitness = float(metrics.get("eval_fitness") or 0.0)
    trimmed = metrics.get("eval_trimmed_mean_nn_dist")
    scale_min = float(metrics.get("scale_min") or 0.0)
    scale_max = float(metrics.get("scale_max") or 0.0)

    if fitness < float(gate.get("min_fitness", 0.55)):
        return "needs_review_low_fitness"
    if trimmed is None or float(trimmed) > float(gate.get("max_trimmed_mean_nn_dist", 0.35)):
        return "needs_review_high_trimmed"
    if scale_min < float(gate.get("min_scale", 0.45)) or scale_max > float(gate.get("max_scale", 2.3)):
        return "needs_review_scale"
    return "accepted_by_numeric_gate"

from __future__ import annotations

import numpy as np
import open3d as o3d


def nearest_neighbor_distances(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    max_points: int = 30000,
) -> np.ndarray:
    """计算 source 到 target 的最近邻距离。

    为了避免大点云评估过慢，默认等距抽样到 `max_points`。
    评估阈值偏严格，因此抽样不会改变“是否明显错误”的判断方向。
    """

    source_points = np.asarray(source.points)
    if len(source_points) == 0 or len(target.points) == 0:
        return np.asarray([], dtype=float)
    if len(source_points) > max_points:
        indices = np.linspace(0, len(source_points) - 1, max_points).astype(int)
        source_points = source_points[indices]

    tree = o3d.geometry.KDTreeFlann(target)
    distances = np.empty(len(source_points), dtype=float)
    for index, point in enumerate(source_points):
        count, _, sq_distances = tree.search_knn_vector_3d(point, 1)
        distances[index] = float(np.sqrt(sq_distances[0])) if count else np.inf
    return distances


def summarize_distances(distances: np.ndarray, threshold: float, trimmed_ratio: float) -> dict[str, float | None]:
    """把最近邻距离转换为严格几何指标。"""

    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return {
            "fitness": 0.0,
            "inlier_rmse": None,
            "median_nn": None,
            "trimmed_mean_nn": None,
        }

    inliers = finite[finite <= threshold]
    trimmed_count = max(1, int(len(finite) * trimmed_ratio))
    trimmed = np.sort(finite)[:trimmed_count]
    return {
        "fitness": float(len(inliers) / len(finite)),
        "inlier_rmse": float(np.sqrt(np.mean(inliers * inliers))) if len(inliers) else None,
        "median_nn": float(np.median(finite)),
        "trimmed_mean_nn": float(np.mean(trimmed)),
    }


def matrix_diagnostics(matrix: np.ndarray) -> dict[str, float]:
    """检查矩阵是否接近刚体。"""

    linear = matrix[:3, :3]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    return {
        "det": float(np.linalg.det(linear)),
        "scale_min": float(np.min(singular_values)),
        "scale_max": float(np.max(singular_values)),
        "translation_norm": float(np.linalg.norm(matrix[:3, 3])),
    }


def evaluate_registration(
    source_transformed: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    *,
    threshold: float = 0.08,
    trimmed_ratio: float = 0.8,
    max_eval_points: int = 30000,
) -> dict[str, float | None]:
    """严格评估配准质量。

    输出包含 source->target 和 target->source 两个方向的 fitness。
    target->source 在 source 很小、target 很大时会偏低；这里仍然保留它，
    目的是提醒“只贴到地图一小块”的解不要被单向 fitness 误判为高质量。
    """

    forward = summarize_distances(
        nearest_neighbor_distances(source_transformed, target, max_points=max_eval_points),
        threshold,
        trimmed_ratio,
    )
    backward = summarize_distances(
        nearest_neighbor_distances(target, source_transformed, max_points=max_eval_points),
        threshold,
        trimmed_ratio,
    )

    source_fitness = float(forward["fitness"] or 0.0)
    target_fitness = float(backward["fitness"] or 0.0)
    if source_fitness + target_fitness > 0:
        harmonic = 2.0 * source_fitness * target_fitness / (source_fitness + target_fitness)
    else:
        harmonic = 0.0

    return {
        "threshold": threshold,
        "trimmed_ratio": trimmed_ratio,
        "source_fitness": source_fitness,
        "target_fitness": target_fitness,
        "bidirectional_fitness_hmean": float(harmonic),
        "source_inlier_rmse": forward["inlier_rmse"],
        "source_median_nn": forward["median_nn"],
        "source_trimmed_mean_nn": forward["trimmed_mean_nn"],
        "target_trimmed_mean_nn": backward["trimmed_mean_nn"],
    }


def strict_score(metrics: dict[str, float | None]) -> float:
    """用于候选排序的偏严格分数。"""

    hmean = float(metrics.get("bidirectional_fitness_hmean") or 0.0)
    source_fit = float(metrics.get("source_fitness") or 0.0)
    trimmed = float(metrics.get("source_trimmed_mean_nn") or 999.0)
    rmse = float(metrics.get("source_inlier_rmse") or 999.0)
    return hmean + 0.35 * source_fit - 0.8 * trimmed - 0.2 * rmse


def gate_status(metrics: dict[str, float | None], gate: dict) -> str:
    """给配准结果打严格状态标签。"""

    if float(metrics.get("source_fitness") or 0.0) < float(gate.get("min_source_fitness", 0.35)):
        return "needs_review_low_source_fitness"
    if float(metrics.get("bidirectional_fitness_hmean") or 0.0) < float(gate.get("min_bidir_fitness_hmean", 0.08)):
        return "needs_review_low_bidirectional_fitness"
    trimmed = metrics.get("source_trimmed_mean_nn")
    if trimmed is None or float(trimmed) > float(gate.get("max_source_trimmed_mean_nn", 0.10)):
        return "needs_review_high_trimmed"
    if float(metrics.get("scale_min") or 0.0) < 0.95 or float(metrics.get("scale_max") or 999.0) > 1.05:
        return "needs_review_non_rigid_matrix"
    return "accepted_by_strict_gate"

from __future__ import annotations

from typing import Any

import numpy as np

from pcr.domain import EvaluationMetrics
from pcr.algorithms.shared_frame.geometry import transform_points


def error_metrics(errors: np.ndarray, threshold: float) -> dict[str, float]:
    """统计对应点误差。"""

    return {
        "inlier_threshold": float(threshold),
        "inlier_count": int(np.count_nonzero(errors <= threshold)),
        "inlier_ratio": float(np.mean(errors <= threshold)) if len(errors) else 0.0,
        "rmse": float(np.sqrt(np.mean(errors * errors))) if len(errors) else float("inf"),
        "median_error": float(np.median(errors)) if len(errors) else float("inf"),
        "p90_error": float(np.percentile(errors, 90)) if len(errors) else float("inf"),
        "mean_error": float(np.mean(errors)) if len(errors) else float("inf"),
    }


def frame_error_metrics(
    source: np.ndarray,
    target: np.ndarray,
    frame_names: np.ndarray,
    matrix: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    """按共享帧拆分误差，便于定位某一帧是否拖累整体估计。"""

    transformed = transform_points(source, matrix)
    errors = np.linalg.norm(transformed - target, axis=1)
    rows: list[dict[str, Any]] = []
    for frame in sorted({str(item) for item in frame_names}):
        mask = frame_names == frame
        frame_errors = errors[mask]
        metrics = error_metrics(frame_errors, threshold)
        rows.append(
            {
                "frame": frame,
                "correspondence_count": int(np.count_nonzero(mask)),
                **metrics,
            }
        )
    return rows


def evaluate_shared_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    frame_names: np.ndarray,
    matrix: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """用统一阈值评估某个矩阵在共享帧对应点上的表现。"""

    errors = np.linalg.norm(transform_points(source, matrix) - target, axis=1)
    metrics: dict[str, Any] = error_metrics(errors, threshold)
    metrics["per_frame_metrics"] = frame_error_metrics(source, target, frame_names, matrix, threshold)
    return metrics


def strict_score(metrics: EvaluationMetrics) -> float:
    """共享帧候选分数。

    与旧算法保持一致：奖励 inlier ratio，惩罚 median 和 p90 误差。
    """

    return float(metrics.inlier_ratio - metrics.median_error - 0.25 * metrics.p90_error)

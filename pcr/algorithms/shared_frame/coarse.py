from __future__ import annotations

from typing import Any

import numpy as np

from pcr.algorithms.shared_frame.geometry import estimate_rigid_transform, transform_points
from pcr.domain import EvaluationMetrics, RegistrationCandidate, Transform
from pcr.evaluation.ranking import score_metrics
from pcr.evaluation.registration_metrics import error_metrics, evaluate_shared_correspondences


def refine_rigid_by_thresholds(
    source: np.ndarray,
    target: np.ndarray,
    matrix: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """在候选矩阵附近做逐阈值刚体 refit。"""

    refinement = config.get("refinement", {})
    if not bool(refinement.get("enabled", True)):
        return matrix, []

    thresholds = [float(item) for item in refinement.get("thresholds", [0.12, 0.10, 0.08, 0.06])]
    iterations_per_threshold = int(refinement.get("iterations_per_threshold", 2))
    min_points = int(refinement.get("min_points", 200))
    current = matrix
    history: list[dict[str, Any]] = []

    for threshold in thresholds:
        for iteration in range(iterations_per_threshold):
            errors = np.linalg.norm(transform_points(source, current) - target, axis=1)
            inliers = errors <= threshold
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count < min_points:
                history.append(
                    {
                        "threshold": float(threshold),
                        "iteration": int(iteration),
                        "status": "skipped_not_enough_inliers",
                        "inlier_count": inlier_count,
                    }
                )
                break

            current = estimate_rigid_transform(source[inliers], target[inliers])
            refined_errors = np.linalg.norm(transform_points(source, current) - target, axis=1)
            step_metrics = error_metrics(refined_errors, threshold)
            history.append(
                {
                    "threshold": float(threshold),
                    "iteration": int(iteration),
                    "status": "refit",
                    **step_metrics,
                }
            )

    return current, history


def ransac_rigid_candidates(
    source: np.ndarray,
    target: np.ndarray,
    frame_names: np.ndarray,
    config: dict[str, Any],
    *,
    source_frame: str = "source",
    target_frame: str = "target",
) -> list[RegistrationCandidate]:
    """多阈值 RANSAC + 逐步收紧 refit，生成一组刚体候选。"""

    ransac = config.get("ransac", {})
    iterations = int(ransac.get("iterations", 2000))
    sample_size = int(ransac.get("sample_size", 6))
    thresholds = [float(item) for item in ransac.get("thresholds", [ransac.get("inlier_threshold", 0.06)])]
    evaluation_threshold = float(ransac.get("evaluation_threshold", ransac.get("inlier_threshold", 0.06)))
    seed = int(ransac.get("seed", 7))
    min_inliers = max(sample_size, int(ransac.get("min_inliers", 50)))
    if len(source) < sample_size:
        raise RuntimeError(f"Not enough correspondences for RANSAC: {len(source)} < {sample_size}")

    rng = np.random.default_rng(seed)
    candidates: list[RegistrationCandidate] = []

    for candidate_id, threshold in enumerate(thresholds, 1):
        best_matrix: np.ndarray | None = None
        best_inliers: np.ndarray | None = None
        best_tuple = (-1, float("inf"), float("inf"))

        for _ in range(iterations):
            sample = rng.choice(len(source), size=sample_size, replace=False)
            matrix = estimate_rigid_transform(source[sample], target[sample])
            errors = np.linalg.norm(transform_points(source, matrix) - target, axis=1)
            inliers = errors <= threshold
            count = int(np.count_nonzero(inliers))
            median = float(np.median(errors[inliers])) if count else float("inf")
            rmse = float(np.sqrt(np.mean(errors[inliers] * errors[inliers]))) if count else float("inf")
            key = (count, -median, -rmse)
            if key > best_tuple:
                best_tuple = key
                best_matrix = matrix
                best_inliers = inliers

        if best_matrix is None or best_inliers is None or np.count_nonzero(best_inliers) < min_inliers:
            continue

        refit = estimate_rigid_transform(source[best_inliers], target[best_inliers])
        refined, refinement_history = refine_rigid_by_thresholds(source, target, refit, config)
        metric_dict = evaluate_shared_correspondences(source, target, frame_names, refined, evaluation_threshold)
        metric_dict.update(
            {
                "coarse_threshold": float(threshold),
                "evaluation_threshold": float(evaluation_threshold),
                "ransac_iterations": iterations,
                "ransac_sample_size": sample_size,
                "refit_inlier_count": int(np.count_nonzero(best_inliers)),
                "refinement_history": refinement_history,
            }
        )
        metrics = EvaluationMetrics.from_mapping(metric_dict)
        candidates.append(
            RegistrationCandidate(
                candidate_id=str(candidate_id),
                transform=Transform(source=source_frame, target=target_frame, matrix=refined),
                score=score_metrics(metric_dict),
                metrics=metrics,
                metadata={"method": "shared_frame_rigid_threshold_refine"},
            )
        )

    if not candidates:
        raise RuntimeError("RANSAC failed to find a valid shared-frame transform.")

    return sorted(candidates, key=lambda item: item.score, reverse=True)


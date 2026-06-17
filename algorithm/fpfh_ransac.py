from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from algorithm.base import RegistrationResult
from algorithm.common import compute_fpfh, evaluate_matrix, prepare_clouds, select_best
from utils.pointcloud import load_cloud


def register(source_path: str | Path, target_path: str | Path, config: dict) -> RegistrationResult:
    """FPFH + RANSAC 粗配准。

    参数偏严格：较小 correspondence distance、较高 edge length checker，
    避免重复平面造成过高的假 fitness。
    """

    source_path = Path(source_path)
    target_path = Path(target_path)
    source = load_cloud(source_path)
    target = load_cloud(target_path)
    source_down, target_down = prepare_clouds(source, target, config)
    source_fpfh = compute_fpfh(source_down, config.get("features", {}))
    target_fpfh = compute_fpfh(target_down, config.get("features", {}))

    ransac_cfg = config.get("ransac", {})
    distance = float(ransac_cfg.get("max_correspondence_distance", 0.10))
    attempts = int(ransac_cfg.get("attempts", 4))
    candidates = []
    for attempt in range(attempts):
        try:
            o3d.utility.random.seed(int(ransac_cfg.get("seed", 7)) + attempt)
        except AttributeError:
            pass
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            bool(ransac_cfg.get("mutual_filter", True)),
            distance,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            int(ransac_cfg.get("ransac_n", 4)),
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    float(ransac_cfg.get("edge_length_checker", 0.95))
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(
                int(ransac_cfg.get("max_iterations", 60000)),
                float(ransac_cfg.get("confidence", 0.999)),
            ),
        )
        matrix = np.asarray(result.transformation, dtype=float)
        if not np.all(np.isfinite(matrix)):
            continue
        candidates.append(
            evaluate_matrix(
                attempt + 1,
                matrix,
                source_down,
                target_down,
                config,
                metadata={
                    "coarse_method": "fpfh_ransac",
                    "open3d_fitness": float(result.fitness),
                    "open3d_inlier_rmse": float(result.inlier_rmse),
                },
            )
        )

    if not candidates:
        raise RuntimeError("FPFH RANSAC produced no valid candidates.")
    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    best = select_best(candidates)
    return RegistrationResult(
        algorithm="fpfh_ransac",
        source_path=source_path,
        target_path=target_path,
        status=str(best.metrics["status"]),
        matrix=best.matrix,
        metrics=best.metrics,
        candidates=candidates[: int(config.get("output", {}).get("top_k", 5))],
    )

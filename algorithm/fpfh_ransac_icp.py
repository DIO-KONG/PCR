from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from utils.metrics import evaluate_registration, matrix_diagnostics, quality_status
from utils.pointcloud import clone_transform, read_point_cloud
from utils.preprocess import prepare_point_cloud
from utils.types import Candidate, RegistrationResult


def register(source_path: str | Path, target_path: str | Path, config: dict) -> RegistrationResult:
    """Open3D FPFH + RANSAC + ICP 对照算法。

    该算法主要用于 benchmark/sanity check。它输出严格刚体矩阵，
    但在 DA3 ghosting、尺度漂移和重复平面场景下可能不如 topview_vote 稳定。
    """

    source_path = Path(source_path)
    target_path = Path(target_path)
    source_raw = read_point_cloud(source_path)
    target_raw = read_point_cloud(target_path)
    preprocess_config = dict(config.get("preprocess", {}))
    registration_config = dict(config.get("registration", {}))
    gate = dict(config.get("gate", {}))

    source_prepared = prepare_point_cloud(source_raw, preprocess_config, compute_fpfh=True)
    target_prepared = prepare_point_cloud(target_raw, preprocess_config, compute_fpfh=True)
    voxel_size = float(preprocess_config.get("voxel_size", registration_config.get("registration_voxel_size", 0.1)))

    distance_threshold = voxel_size * float(registration_config.get("ransac_distance_factor", 1.5))
    ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_prepared.down,
        target_prepared.down,
        source_prepared.fpfh,
        target_prepared.fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            int(registration_config.get("ransac_max_iterations", 100000)),
            float(registration_config.get("ransac_confidence", 0.999)),
        ),
    )

    icp_threshold = voxel_size * float(registration_config.get("icp_distance_factor", 1.0))
    icp = o3d.pipelines.registration.registration_icp(
        source_prepared.down,
        target_prepared.down,
        icp_threshold,
        ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(registration_config.get("icp_max_iterations", 50))
        ),
    )

    matrix = np.asarray(icp.transformation, dtype=float)
    transformed_down = clone_transform(source_prepared.down, matrix)
    metrics = {
        **evaluate_registration(
            transformed_down,
            target_prepared.down,
            overlap_threshold=float(registration_config["eval_overlap_threshold"]),
            trimmed_ratio=float(registration_config["eval_trimmed_ratio"]),
        ),
        **matrix_diagnostics(matrix),
        "ransac_fitness": float(ransac.fitness),
        "ransac_inlier_rmse": float(ransac.inlier_rmse),
        "icp_fitness": float(icp.fitness),
        "icp_inlier_rmse": float(icp.inlier_rmse),
    }
    status = quality_status(metrics, gate)
    candidate = Candidate(
        candidate_id=1,
        matrix=matrix,
        coarse_score=float(ransac.fitness),
        metrics=metrics,
        rank_score=float(metrics.get("eval_fitness") or 0.0),
        metadata={"status_hint": status},
    )
    return RegistrationResult(
        algorithm="fpfh_ransac_icp",
        source_path=source_path,
        target_path=target_path,
        matrix=matrix,
        metrics={**metrics, "status": status},
        status=status,
        top_candidates=[candidate],
    )

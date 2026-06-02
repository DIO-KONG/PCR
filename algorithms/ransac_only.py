from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from algorithms.base import RegistrationResult
from algorithms.common import correspondence_count, merged_params, require_source_to_target_matrix


METHOD_NAME = "ransac_only"

DEFAULT_PARAMS = {
    "voxel_size": 0.05,
    "distance_threshold_factor": 1.5,
    "ransac_n": 4,
    "max_iteration": 100000,
    "confidence": 0.999,
}


def run(prepared_source: Any, prepared_target: Any, params: Mapping[str, Any] | None = None) -> RegistrationResult:
    used_params = merged_params(DEFAULT_PARAMS, params)
    started = time.perf_counter()

    try:
        import open3d as o3d
    except ImportError:
        return RegistrationResult.skipped(METHOD_NAME, "Missing optional dependency: open3d.", used_params)

    try:
        source_pcd = prepared_source["pcd_down"]
        target_pcd = prepared_target["pcd_down"]
        source_fpfh = prepared_source["fpfh"]
        target_fpfh = prepared_target["fpfh"]

        voxel_size = float(used_params["voxel_size"])
        distance_threshold = voxel_size * float(used_params["distance_threshold_factor"])

        criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
            int(used_params["max_iteration"]),
            float(used_params["confidence"]),
        )
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd,
            target_pcd,
            source_fpfh,
            target_fpfh,
            True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            int(used_params["ransac_n"]),
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            criteria,
        )

        transformation = require_source_to_target_matrix(result.transformation)
        status = "success" if np.isfinite(transformation).all() else "failed"
        return RegistrationResult(
            method=METHOD_NAME,
            status=status,
            transformation=transformation,
            runtime_sec=time.perf_counter() - started,
            fitness=float(result.fitness),
            inlier_rmse=float(result.inlier_rmse),
            correspondence_set_size=correspondence_count(result),
            params=used_params,
        )
    except Exception as exc:
        return RegistrationResult.failed(METHOD_NAME, str(exc), used_params)

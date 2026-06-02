from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np

from algorithms.base import RegistrationResult
from algorithms.common import correspondence_count, merged_params, require_source_to_target_matrix


METHOD_NAME = "gicp_refine"

DEFAULT_PARAMS = {
    "voxel_size": 0.8,
    "distance_threshold_factor": 2.0,
    "ransac_n": 4,
    "ransac_trials": 5,
    "max_iteration": 100000,
    "confidence": 0.999,
    "gicp_max_iterations": 20,
    "gicp_max_correspondence_distance_factor": 2.0,
    "gicp_downsampling_resolution": 0.8,
    "gicp_num_threads": 1,
    "gicp_verbose": False,
    "enable_refine": True,
}


def run(prepared_source: Any, prepared_target: Any, params: Mapping[str, Any] | None = None) -> RegistrationResult:
    used_params = merged_params(DEFAULT_PARAMS, params)
    started = time.perf_counter()

    try:
        import open3d as o3d
    except ImportError:
        return RegistrationResult.skipped(METHOD_NAME, "Missing required dependency for coarse registration: open3d.", used_params)

    try:
        import small_gicp
    except ImportError:
        return RegistrationResult.skipped(
            METHOD_NAME,
            "Missing required dependency: small_gicp. Install with `.env\\python.exe -m pip install small-gicp`.",
            used_params,
        )

    coarse_started = time.perf_counter()
    all_trials: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    best_result = None

    try:
        source_pcd = prepared_source["pcd_down"]
        target_pcd = prepared_target["pcd_down"]
        source_fpfh = prepared_source["fpfh"]
        target_fpfh = prepared_target["fpfh"]

        voxel_size = float(used_params["voxel_size"])
        distance_threshold = voxel_size * float(used_params["distance_threshold_factor"])
        ransac_trials = max(1, int(used_params["ransac_trials"]))

        criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
            int(used_params["max_iteration"]),
            float(used_params["confidence"]),
        )

        for trial_idx in range(ransac_trials):
            trial_started = time.perf_counter()
            coarse_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
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
            corr_count = correspondence_count(coarse_result)
            trial_record = {
                "trial": trial_idx + 1,
                "fitness": float(coarse_result.fitness),
                "inlier_rmse": float(coarse_result.inlier_rmse),
                "correspondence_set_size": corr_count,
                "time": time.perf_counter() - trial_started,
            }
            all_trials.append(trial_record)
            if _is_better_trial(trial_record, best_trial):
                best_trial = trial_record
                best_result = coarse_result

        coarse_time = time.perf_counter() - coarse_started
        if best_result is None or best_trial is None or float(best_trial["fitness"]) <= 0.0:
            return RegistrationResult(
                method=METHOD_NAME,
                status="failed",
                transformation=None,
                runtime_sec=time.perf_counter() - started,
                params=used_params,
                error="RANSAC produced no valid coarse transform for GICP refinement.",
                algorithm_metrics={
                    "ransac_trials": ransac_trials,
                    "best_trial": None,
                    "best_coarse_fitness": None,
                    "best_coarse_inlier_rmse": None,
                    "all_trials": all_trials,
                    "coarse_time": coarse_time,
                    "refine_time": 0.0,
                },
            )

        init_transform = require_source_to_target_matrix(best_result.transformation)
        if not bool(used_params["enable_refine"]):
            return RegistrationResult(
                method=METHOD_NAME,
                status="success",
                transformation=init_transform,
                runtime_sec=time.perf_counter() - started,
                fitness=float(best_trial["fitness"]),
                inlier_rmse=float(best_trial["inlier_rmse"]),
                correspondence_set_size=best_trial["correspondence_set_size"],
                params=used_params,
                algorithm_metrics={
                    "ransac_trials": ransac_trials,
                    "best_trial": best_trial["trial"],
                    "best_coarse_fitness": float(best_trial["fitness"]),
                    "best_coarse_inlier_rmse": float(best_trial["inlier_rmse"]),
                    "all_trials": all_trials,
                    "gicp_converged": None,
                    "gicp_error": None,
                    "gicp_iterations": None,
                    "gicp_num_inliers": None,
                    "coarse_time": coarse_time,
                    "refine_time": 0.0,
                    "refine_enabled": False,
                },
            )

        refine_started = time.perf_counter()
        source_points = np.asarray(source_pcd.points, dtype=np.float64)
        target_points = np.asarray(target_pcd.points, dtype=np.float64)
        gicp_result = small_gicp.align(
            target_points,
            source_points,
            init_transform,
            registration_type="GICP",
            downsampling_resolution=float(used_params["gicp_downsampling_resolution"]),
            max_correspondence_distance=voxel_size * float(used_params["gicp_max_correspondence_distance_factor"]),
            num_threads=int(used_params["gicp_num_threads"]),
            max_iterations=int(used_params["gicp_max_iterations"]),
            verbose=bool(used_params["gicp_verbose"]),
        )
        refine_time = time.perf_counter() - refine_started
        refined_transform = require_source_to_target_matrix(gicp_result.T_target_source)

        gicp_converged = _safe_attr(gicp_result, "converged")
        gicp_error = _safe_attr(gicp_result, "error")
        gicp_iterations = _safe_attr(gicp_result, "iterations")
        gicp_num_inliers = _safe_attr(gicp_result, "num_inliers")
        status = "success" if gicp_num_inliers is None or int(gicp_num_inliers) > 0 else "failed"
        error = None if status == "success" else "small_gicp produced no inliers during refinement."

        return RegistrationResult(
            method=METHOD_NAME,
            status=status,
            transformation=refined_transform if status == "success" else None,
            runtime_sec=time.perf_counter() - started,
            fitness=float(best_trial["fitness"]),
            inlier_rmse=float(best_trial["inlier_rmse"]),
            correspondence_set_size=best_trial["correspondence_set_size"],
            params=used_params,
            error=error,
            algorithm_metrics={
                "ransac_trials": ransac_trials,
                "best_trial": best_trial["trial"],
                "best_coarse_fitness": float(best_trial["fitness"]),
                "best_coarse_inlier_rmse": float(best_trial["inlier_rmse"]),
                "all_trials": all_trials,
                "gicp_converged": gicp_converged,
                "gicp_error": gicp_error,
                "gicp_iterations": gicp_iterations,
                "gicp_num_inliers": gicp_num_inliers,
                "coarse_time": coarse_time,
                "refine_time": refine_time,
                "refine_enabled": True,
            },
        )
    except Exception as exc:
        return RegistrationResult(
            method=METHOD_NAME,
            status="failed",
            transformation=None,
            runtime_sec=time.perf_counter() - started,
            params=used_params,
            error=str(exc),
            algorithm_metrics={
                "ransac_trials": int(used_params.get("ransac_trials", DEFAULT_PARAMS["ransac_trials"])),
                "best_trial": best_trial["trial"] if best_trial else None,
                "best_coarse_fitness": float(best_trial["fitness"]) if best_trial else None,
                "best_coarse_inlier_rmse": float(best_trial["inlier_rmse"]) if best_trial else None,
                "all_trials": all_trials,
                "coarse_time": time.perf_counter() - coarse_started,
                "refine_time": 0.0,
            },
        )


def _is_better_trial(candidate: Mapping[str, Any], current: Mapping[str, Any] | None) -> bool:
    if current is None:
        return True
    candidate_score = (float(candidate["fitness"]), -float(candidate["inlier_rmse"]))
    current_score = (float(current["fitness"]), -float(current["inlier_rmse"]))
    return candidate_score > current_score


def _safe_attr(obj: Any, name: str) -> Any:
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value

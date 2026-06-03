from __future__ import annotations

import itertools
import time
from typing import Any, Mapping

import numpy as np

from algorithms.base import RegistrationResult
from algorithms.common import correspondence_count, merged_params, require_source_to_target_matrix
from utils import metrics


METHOD_NAME = "multi_comb"

DEFAULT_PARAMS = {
    "voxel_size": 0.8,
    "coarse_method": "ransac",
    "coarse_trials": 3,
    "distance_threshold_factor": 2.0,
    "ransac_n": 4,
    "max_iteration": 100000,
    "confidence": 0.999,
    "affine_mode": "none",
    "scale_min": 0.8,
    "scale_max": 1.2,
    "scale_steps": 5,
    "inlier_distance_factor": 2.0,
    "score_weights": {
        "w_fitness": 1.0,
        "w_overlap": 0.5,
        "w_rmse": 0.5,
        "w_trimmed": 0.5,
        "w_translation": 0.0,
    },
}


def run(prepared_source: Any, prepared_target: Any, params: Mapping[str, Any] | None = None) -> RegistrationResult:
    used_params = merged_params(DEFAULT_PARAMS, params)
    used_params["score_weights"] = dict(DEFAULT_PARAMS["score_weights"]) | dict(used_params.get("score_weights", {}))
    started = time.perf_counter()

    try:
        import open3d as o3d
    except ImportError:
        return RegistrationResult.skipped(METHOD_NAME, "Missing required dependency: open3d.", used_params)

    coarse_started = time.perf_counter()
    all_trials: list[dict[str, Any]] = []
    best_trial: dict[str, Any] | None = None
    best_transform: np.ndarray | None = None
    source_pcd = prepared_source["pcd_down"]
    target_pcd = prepared_target["pcd_down"]

    try:
        coarse_method = str(used_params.get("coarse_method", "ransac")).lower()
        coarse_trials = max(1, int(used_params.get("coarse_trials", 3)))
        if coarse_method not in {"ransac", "fgr"}:
            return RegistrationResult.skipped(METHOD_NAME, f"Unsupported coarse_method: {coarse_method}", used_params)

        for trial_idx in range(1, coarse_trials + 1):
            trial = _run_coarse_trial(o3d, prepared_source, prepared_target, used_params, coarse_method, trial_idx)
            if trial["status"] == "success":
                eval_metrics = metrics.evaluate_registration(
                    source_pcd,
                    target_pcd,
                    trial["transformation"],
                    {"trimmed_ratio": 0.9, "overlap_threshold": float(used_params["voxel_size"]) * float(used_params["distance_threshold_factor"])},
                )
                trial.update(eval_metrics)
                score, warnings = _score_trial(eval_metrics, used_params["score_weights"])
                trial["score"] = score
                if warnings:
                    trial["warnings"] = warnings
            else:
                trial["score"] = None
            trial_summary = {k: v for k, v in trial.items() if k != "transformation"}
            all_trials.append(trial_summary)
            if trial["status"] == "success" and _is_better_candidate(trial, best_trial):
                best_trial = trial_summary
                best_transform = trial["transformation"]

        coarse_time = time.perf_counter() - coarse_started
        if best_trial is None or best_transform is None:
            return RegistrationResult(
                method=METHOD_NAME,
                status="failed",
                transformation=None,
                runtime_sec=time.perf_counter() - started,
                params=used_params,
                error="No valid coarse registration candidate.",
                algorithm_metrics=_metrics_payload(used_params, None, None, all_trials, coarse_time, 0.0, "none", None, "no valid coarse candidate"),
            )

        affine_started = time.perf_counter()
        final_transform = best_transform
        affine_error = None
        scale_values = None
        affine_mode = str(used_params.get("affine_mode", "none")).lower()
        transform_type = "rigid"
        if affine_mode != "none":
            try:
                final_transform, scale_values, transform_type = _affine_refine(source_pcd, target_pcd, best_transform, used_params, affine_mode)
            except Exception as exc:
                affine_error = str(exc)
                transform_type = "rigid_fallback"
                final_transform = best_transform
        affine_time = time.perf_counter() - affine_started

        return RegistrationResult(
            method=METHOD_NAME,
            status="success",
            transformation=require_source_to_target_matrix(final_transform),
            runtime_sec=time.perf_counter() - started,
            fitness=best_trial.get("algorithm_fitness"),
            inlier_rmse=best_trial.get("algorithm_inlier_rmse"),
            params=used_params,
            algorithm_metrics=_metrics_payload(
                used_params,
                best_trial,
                best_trial.get("score"),
                all_trials,
                coarse_time,
                affine_time,
                transform_type,
                scale_values,
                affine_error,
            ),
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
                "coarse_method": used_params.get("coarse_method"),
                "coarse_trials": used_params.get("coarse_trials"),
                "all_trials": all_trials,
                "coarse_time": time.perf_counter() - coarse_started,
                "affine_time": 0.0,
                "affine_error": str(exc),
            },
        )


def _run_coarse_trial(o3d: Any, prepared_source: Mapping[str, Any], prepared_target: Mapping[str, Any], params: Mapping[str, Any], method: str, trial_id: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        if method == "ransac":
            result = _run_ransac(o3d, prepared_source, prepared_target, params)
        else:
            result = _run_fgr(o3d, prepared_source, prepared_target, params)
        transform = require_source_to_target_matrix(result.transformation)
        corr_count = correspondence_count(result)
        if float(result.fitness) <= 0.0 or (corr_count is not None and corr_count <= 0):
            return _failed_trial(trial_id, method, "coarse candidate has no valid correspondences", started)
        return {
            "trial_id": trial_id,
            "coarse_method": method,
            "status": "success",
            "algorithm_fitness": float(result.fitness),
            "algorithm_inlier_rmse": float(result.inlier_rmse),
            "algorithm_correspondence_set_size": corr_count,
            "transformation": transform,
            "time": time.perf_counter() - started,
        }
    except Exception as exc:
        return _failed_trial(trial_id, method, str(exc), started)


def _run_ransac(o3d: Any, prepared_source: Mapping[str, Any], prepared_target: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    voxel_size = float(params["voxel_size"])
    distance_threshold = voxel_size * float(params["distance_threshold_factor"])
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        prepared_source["pcd_down"],
        prepared_target["pcd_down"],
        prepared_source["fpfh"],
        prepared_target["fpfh"],
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        int(params["ransac_n"]),
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(int(params["max_iteration"]), float(params["confidence"])),
    )


def _run_fgr(o3d: Any, prepared_source: Mapping[str, Any], prepared_target: Mapping[str, Any], params: Mapping[str, Any]) -> Any:
    registration = o3d.pipelines.registration
    if not hasattr(registration, "registration_fgr_based_on_feature_matching"):
        raise RuntimeError("Current Open3D build does not provide registration_fgr_based_on_feature_matching.")
    voxel_size = float(params["voxel_size"])
    option = registration.FastGlobalRegistrationOption(maximum_correspondence_distance=voxel_size * float(params["distance_threshold_factor"]))
    return registration.registration_fgr_based_on_feature_matching(
        prepared_source["pcd_down"],
        prepared_target["pcd_down"],
        prepared_source["fpfh"],
        prepared_target["fpfh"],
        option,
    )


def _failed_trial(trial_id: int, method: str, error: str, started: float) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "coarse_method": method,
        "status": "failed",
        "error": error,
        "time": time.perf_counter() - started,
    }


def _score_trial(eval_metrics: Mapping[str, Any], weights: Mapping[str, float]) -> tuple[float, list[str]]:
    score = 0.0
    warnings: list[str] = []
    terms = [
        ("eval_fitness", "w_fitness", 1.0),
        ("eval_overlap_ratio", "w_overlap", 1.0),
        ("eval_inlier_rmse", "w_rmse", -1.0),
        ("eval_trimmed_mean_nn_dist", "w_trimmed", -1.0),
        ("eval_translation_norm", "w_translation", -1.0),
    ]
    for metric_key, weight_key, sign in terms:
        value = eval_metrics.get(metric_key)
        if value is None:
            warnings.append(f"missing {metric_key}")
            continue
        score += float(weights.get(weight_key, 0.0)) * sign * float(value)
    return score, warnings


def _is_better_candidate(candidate: Mapping[str, Any], current: Mapping[str, Any] | None) -> bool:
    if current is None:
        return True
    return float(candidate.get("score") or float("-inf")) > float(current.get("score") or float("-inf"))


def _affine_refine(source_pcd: Any, target_pcd: Any, coarse_transform: np.ndarray, params: Mapping[str, Any], affine_mode: str) -> tuple[np.ndarray, dict[str, float] | None, str]:
    source_points, target_points = _nearest_inlier_pairs(source_pcd, target_pcd, coarse_transform, params)
    if len(source_points) < 4:
        raise RuntimeError("Not enough inlier correspondences for affine refine.")
    if affine_mode == "constrained":
        return _constrained_affine(source_points, target_points, coarse_transform, params)
    if affine_mode == "unconstrained":
        return _unconstrained_affine(source_points, target_points)
    raise RuntimeError(f"Unsupported affine_mode: {affine_mode}")


def _nearest_inlier_pairs(source_pcd: Any, target_pcd: Any, transform: np.ndarray, params: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    source_points = np.asarray(source_pcd.points, dtype=float)
    transformed = (transform[:3, :3] @ source_points.T).T + transform[:3, 3]
    target_points = np.asarray(target_pcd.points, dtype=float)
    tree = o3d.geometry.KDTreeFlann(target_pcd)
    max_dist = float(params["voxel_size"]) * float(params.get("inlier_distance_factor", 2.0))
    src_pairs = []
    tgt_pairs = []
    for source_point, transformed_point in zip(source_points, transformed):
        count, idx, squared = tree.search_knn_vector_3d(transformed_point, 1)
        if count and float(np.sqrt(squared[0])) <= max_dist:
            src_pairs.append(source_point)
            tgt_pairs.append(target_points[idx[0]])
    return np.asarray(src_pairs, dtype=float), np.asarray(tgt_pairs, dtype=float)


def _constrained_affine(source_points: np.ndarray, target_points: np.ndarray, coarse_transform: np.ndarray, params: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, float], str]:
    rotation = coarse_transform[:3, :3]
    best_error = float("inf")
    best_transform = None
    best_scale = None
    scales = np.linspace(float(params["scale_min"]), float(params["scale_max"]), int(params["scale_steps"]))
    for sx, sy, sz in itertools.product(scales, repeat=3):
        scale = np.diag([sx, sy, sz])
        rotated_scaled = (rotation @ scale @ source_points.T).T
        translation = np.mean(target_points - rotated_scaled, axis=0)
        residual = rotated_scaled + translation - target_points
        error = float(np.mean(np.sum(residual * residual, axis=1)))
        if error < best_error:
            best_error = error
            best_transform = np.eye(4)
            best_transform[:3, :3] = rotation @ scale
            best_transform[:3, 3] = translation
            best_scale = {"sx": float(sx), "sy": float(sy), "sz": float(sz)}
    if best_transform is None or best_scale is None:
        raise RuntimeError("Constrained affine grid search failed.")
    return best_transform, best_scale, "constrained_affine"


def _unconstrained_affine(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, None, str]:
    design = np.hstack([source_points, np.ones((len(source_points), 1))])
    solution, _, _, _ = np.linalg.lstsq(design, target_points, rcond=None)
    transform = np.eye(4)
    transform[:3, :3] = solution[:3, :].T
    transform[:3, 3] = solution[3, :]
    return transform, None, "unconstrained_affine"


def _metrics_payload(
    params: Mapping[str, Any],
    best_trial: Mapping[str, Any] | None,
    best_score: float | None,
    all_trials: list[dict[str, Any]],
    coarse_time: float,
    affine_time: float,
    transform_type: str,
    scale_values: dict[str, float] | None,
    affine_error: str | None,
) -> dict[str, Any]:
    payload = {
        "coarse_method": params.get("coarse_method"),
        "coarse_trials": int(params.get("coarse_trials", 3)),
        "best_trial": best_trial.get("trial_id") if best_trial else None,
        "best_score": best_score,
        "all_trials": all_trials,
        "affine_mode": params.get("affine_mode"),
        "transform_type": transform_type,
        "scale_values": scale_values,
        "affine_error": affine_error,
        "coarse_time": coarse_time,
        "affine_time": affine_time,
    }
    if affine_error:
        payload["affine_failed"] = True
    return payload

from __future__ import annotations

import itertools
import time
from typing import Any, Mapping

import numpy as np

from algorithms.base import RegistrationResult
from algorithms.common import correspondence_count, merged_params, require_source_to_target_matrix
from utils import coverage, degeneracy, metrics, point_filters, pose_prior


METHOD_NAME = "multi_comb"

DEFAULT_PARAMS = {
    "voxel_size": 0.8,
    "coarse_method": "ransac",
    "coarse_methods": ["ransac", "fgr"],
    "coarse_trials": 2,
    "voxel_sizes": [0.8],
    "distance_threshold_factors": [2.0],
    "feature_subsets": ["all", "non_floor"],
    "distance_threshold_factor": 2.0,
    "ransac_n": 4,
    "max_iteration": 100000,
    "confidence": 0.999,
    "top_k_refine": 2,
    "min_rotation_separation_deg": 5.0,
    "min_translation_separation": 0.3,
    "rigid_refine": {
        "enabled": True,
        "method": "gicp",
        "max_iterations": 20,
        "max_correspondence_distance_factor": 1.5,
    },
    "affine_mode": "none",
    "scale_min": 0.8,
    "scale_max": 1.2,
    "scale_steps": 5,
    "horizontal_scale_min": None,
    "horizontal_scale_max": None,
    "horizontal_scale_steps": None,
    "vertical_scale_min": None,
    "vertical_scale_max": None,
    "vertical_scale_steps": None,
    "inlier_distance_factor": 2.0,
    "affine_pair_distance_factor": 5.0,
    "affine_trimmed_ratio": 0.7,
    "score_weights": {
        "w_fitness": 1.0,
        "w_overlap": 0.5,
        "w_rmse": 0.5,
        "w_trimmed": 0.5,
        "w_translation": 0.0,
        "w_coverage": 0.8,
        "w_degeneracy": 0.3,
        "w_motion_prior": 0.3,
    },
    "filtering": {
        "floor_band": 0.35,
        "near_mid_radius": 10.0,
        "curvature_knn": 20,
        "curvature_quantile": 0.75,
        "min_subset_points": 50,
    },
    "coverage": {
        "coverage_grid_size": 1.0,
        "coverage_max_bins_per_axis": 64,
    },
    "motion_prior": {
        "enabled": False,
    },
}


def run(prepared_source: Any, prepared_target: Any, params: Mapping[str, Any] | None = None) -> RegistrationResult:
    global _candidate_runtime_records, _feature_cache
    _candidate_runtime_records = []
    _feature_cache = {}
    used_params = merged_params(DEFAULT_PARAMS, params)
    used_params = _merge_nested_defaults(used_params)
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
        candidate_specs = _candidate_specs(used_params)
        for candidate_id, spec in enumerate(candidate_specs, start=1):
            trial = _run_coarse_trial(o3d, prepared_source, prepared_target, used_params, spec, candidate_id)
            if trial["status"] == "success":
                _evaluate_and_score_trial(trial, source_pcd, target_pcd, used_params)
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
                algorithm_metrics=_metrics_payload(used_params, None, None, all_trials, coarse_time, 0.0, 0.0, [], "none", None, "no valid coarse candidate"),
            )

        refine_started = time.perf_counter()
        selected = _select_diverse_candidates(
            [trial for trial in all_trials if trial.get("status") == "success"],
            int(used_params.get("top_k_refine", 3)),
            float(used_params.get("min_rotation_separation_deg", 5.0)),
            float(used_params.get("min_translation_separation", 0.3)),
        )
        selected_ids = {trial["candidate_id"] for trial in selected}
        refined_trials: list[dict[str, Any]] = []
        transform_by_candidate = {trial["candidate_id"]: trial["transformation"] for trial in _candidate_runtime_records}
        if dict(used_params.get("rigid_refine", {})).get("enabled", True):
            for selected_trial in selected:
                runtime_trial = transform_by_candidate.get(selected_trial["candidate_id"])
                if runtime_trial is None:
                    continue
                refined = _refine_rigid(o3d, source_pcd, target_pcd, runtime_trial, selected_trial, used_params)
                _evaluate_and_score_trial(refined, source_pcd, target_pcd, used_params)
                refined_trials.append({k: v for k, v in refined.items() if k != "transformation"})
                if refined["status"] == "success" and _is_better_candidate(refined, best_trial):
                    best_trial = {k: v for k, v in refined.items() if k != "transformation"}
                    best_transform = refined["transformation"]
        refine_time = time.perf_counter() - refine_started
        for trial in all_trials:
            trial["selected_for_refine"] = trial.get("candidate_id") in selected_ids

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
                refine_time,
                refined_trials,
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
                "refine_time": 0.0,
                "affine_time": 0.0,
                "affine_error": str(exc),
            },
        )


_candidate_runtime_records: list[dict[str, Any]] = []
_feature_cache: dict[tuple[str, float, str], dict[str, Any]] = {}


def _run_coarse_trial(o3d: Any, prepared_source: Mapping[str, Any], prepared_target: Mapping[str, Any], params: Mapping[str, Any], spec: Mapping[str, Any], candidate_id: int) -> dict[str, Any]:
    started = time.perf_counter()
    method = str(spec["coarse_method"])
    try:
        local_params = dict(params)
        local_params["voxel_size"] = spec["voxel_size"]
        local_params["distance_threshold_factor"] = spec["distance_threshold_factor"]
        local_prepared_source, local_prepared_target = _prepare_candidate_features(o3d, prepared_source, prepared_target, spec, params)
        if method == "ransac":
            result = _run_ransac(o3d, local_prepared_source, local_prepared_target, local_params)
        else:
            result = _run_fgr(o3d, local_prepared_source, local_prepared_target, local_params)
        transform = require_source_to_target_matrix(result.transformation)
        corr_count = correspondence_count(result)
        if float(result.fitness) <= 0.0 or (corr_count is not None and corr_count <= 0):
            return _failed_trial(candidate_id, spec, "coarse candidate has no valid correspondences", started)
        trial = {
            "candidate_id": candidate_id,
            "trial_id": spec["trial_id"],
            "coarse_method": method,
            "feature_subset": spec["feature_subset"],
            "voxel_size": spec["voxel_size"],
            "distance_threshold_factor": spec["distance_threshold_factor"],
            "status": "success",
            "algorithm_fitness": float(result.fitness),
            "algorithm_inlier_rmse": float(result.inlier_rmse),
            "algorithm_correspondence_set_size": corr_count,
            "transformation": transform,
            "time": time.perf_counter() - started,
        }
        _candidate_runtime_records.append({"candidate_id": candidate_id, "transformation": transform})
        return trial
    except Exception as exc:
        return _failed_trial(candidate_id, spec, str(exc), started)


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


def _merge_nested_defaults(params: dict[str, Any]) -> dict[str, Any]:
    for key in ("score_weights", "filtering", "coverage", "motion_prior", "rigid_refine"):
        params[key] = dict(DEFAULT_PARAMS[key]) | dict(params.get(key, {}))
    return params


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _candidate_specs(params: Mapping[str, Any]) -> list[dict[str, Any]]:
    methods = [str(method).lower() for method in _as_list(params.get("coarse_methods") or params.get("coarse_method", "ransac"))]
    voxel_sizes = [float(v) for v in _as_list(params.get("voxel_sizes") or params.get("voxel_size", 0.8))]
    distance_factors = [float(v) for v in _as_list(params.get("distance_threshold_factors") or params.get("distance_threshold_factor", 2.0))]
    subsets = [str(v) for v in _as_list(params.get("feature_subsets") or "all")]
    trials = max(1, int(params.get("coarse_trials", 3)))
    specs = []
    for method, voxel_size, distance_factor, subset in itertools.product(methods, voxel_sizes, distance_factors, subsets):
        for trial_id in range(1, trials + 1):
            specs.append(
                {
                    "coarse_method": method,
                    "voxel_size": voxel_size,
                    "distance_threshold_factor": distance_factor,
                    "feature_subset": subset,
                    "trial_id": trial_id,
                }
            )
    return specs


def _prepare_candidate_features(o3d: Any, prepared_source: Mapping[str, Any], prepared_target: Mapping[str, Any], spec: Mapping[str, Any], params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    subset = str(spec.get("feature_subset", "all"))
    voxel_size = float(spec["voxel_size"])
    source_key = ("source", voxel_size, subset)
    target_key = ("target", voxel_size, subset)
    if source_key in _feature_cache and target_key in _feature_cache:
        return _feature_cache[source_key], _feature_cache[target_key]
    source_raw = prepared_source.get("pcd") or prepared_source["pcd_down"]
    target_raw = prepared_target.get("pcd") or prepared_target["pcd_down"]
    filtering_config = dict(params.get("filtering", {}))
    source_subset = point_filters.subset_point_cloud(source_raw, subset, filtering_config)
    target_subset = point_filters.subset_point_cloud(target_raw, subset, filtering_config)
    feature_config = {
        "normal_radius_factor": params.get("normal_radius_factor", 2.0),
        "fpfh_radius_factor": params.get("fpfh_radius_factor", 5.0),
        "normal_max_nn": params.get("normal_max_nn", 30),
        "fpfh_max_nn": params.get("fpfh_max_nn", 100),
    }
    source_features = point_filters.prepare_subset_features(o3d, source_subset, voxel_size, feature_config)
    target_features = point_filters.prepare_subset_features(o3d, target_subset, voxel_size, feature_config)
    _feature_cache[source_key] = source_features
    _feature_cache[target_key] = target_features
    return source_features, target_features


def _evaluate_and_score_trial(trial: dict[str, Any], source_pcd: Any, target_pcd: Any, params: Mapping[str, Any]) -> None:
    overlap_threshold = float(trial.get("voxel_size", params["voxel_size"])) * float(trial.get("distance_threshold_factor", params["distance_threshold_factor"]))
    eval_metrics = metrics.evaluate_registration(source_pcd, target_pcd, trial["transformation"], {"trimmed_ratio": 0.9, "overlap_threshold": overlap_threshold})
    src_inliers, tgt_inliers, _ = coverage.inlier_pairs(source_pcd, target_pcd, trial["transformation"], overlap_threshold)
    coverage_score = coverage.coverage_score(tgt_inliers, np.asarray(target_pcd.points), params.get("coverage", {}))
    plane_degeneracy = degeneracy.plane_degeneracy(tgt_inliers)
    motion_error = pose_prior.motion_prior_error(trial["transformation"], params.get("motion_prior", {}))
    eval_metrics.update(
        {
            "coverage_score": coverage_score,
            "plane_degeneracy": plane_degeneracy,
            "motion_prior_error": motion_error,
            "inlier_count": int(len(tgt_inliers)),
        }
    )
    trial.update(eval_metrics)
    score, warnings = _score_trial(eval_metrics, params["score_weights"])
    trial["score"] = score
    if warnings:
        trial["warnings"] = warnings


def _failed_trial(candidate_id: int, spec: Mapping[str, Any], error: str, started: float) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "trial_id": spec.get("trial_id"),
        "coarse_method": spec.get("coarse_method"),
        "feature_subset": spec.get("feature_subset"),
        "voxel_size": spec.get("voxel_size"),
        "distance_threshold_factor": spec.get("distance_threshold_factor"),
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
        ("coverage_score", "w_coverage", 1.0),
        ("plane_degeneracy", "w_degeneracy", -1.0),
        ("motion_prior_error", "w_motion_prior", -1.0),
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


def _select_diverse_candidates(candidates: list[dict[str, Any]], top_k: int, min_rotation_deg: float, min_translation: float) -> list[dict[str, Any]]:
    ranked = sorted(candidates, key=lambda item: float(item.get("score") or float("-inf")), reverse=True)
    selected: list[dict[str, Any]] = []
    transform_by_candidate = {trial["candidate_id"]: trial["transformation"] for trial in _candidate_runtime_records}
    for candidate in ranked:
        transform = transform_by_candidate.get(candidate["candidate_id"])
        if transform is None:
            continue
        if all(_is_transform_diverse(transform, transform_by_candidate.get(other["candidate_id"]), min_rotation_deg, min_translation) for other in selected):
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected


def _is_transform_diverse(transform: np.ndarray, other: np.ndarray | None, min_rotation_deg: float, min_translation: float) -> bool:
    if other is None:
        return True
    translation_diff = float(np.linalg.norm(transform[:3, 3] - other[:3, 3]))
    rotation_diff = _rotation_angle_deg(transform[:3, :3] @ other[:3, :3].T)
    return translation_diff >= min_translation or rotation_diff >= min_rotation_deg


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) / 2.0
    value = min(1.0, max(-1.0, value))
    return float(np.degrees(np.arccos(value)))


def _refine_rigid(o3d: Any, source_pcd: Any, target_pcd: Any, transform: np.ndarray, parent_trial: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    refine_config = dict(params.get("rigid_refine", {}))
    max_corr = float(parent_trial.get("voxel_size") or params["voxel_size"]) * float(refine_config.get("max_correspondence_distance_factor", 1.5))
    registration = o3d.pipelines.registration
    try:
        if str(refine_config.get("method", "gicp")).lower() == "gicp" and hasattr(registration, "registration_generalized_icp"):
            result = registration.registration_generalized_icp(
                source_pcd,
                target_pcd,
                max_corr,
                transform,
                registration.TransformationEstimationForGeneralizedICP(),
                registration.ICPConvergenceCriteria(max_iteration=int(refine_config.get("max_iterations", 20))),
            )
        else:
            result = registration.registration_icp(
                source_pcd,
                target_pcd,
                max_corr,
                transform,
                registration.TransformationEstimationPointToPlane(),
                registration.ICPConvergenceCriteria(max_iteration=int(refine_config.get("max_iterations", 20))),
            )
        refined = {
            "candidate_id": f"refined_{parent_trial.get('candidate_id')}",
            "parent_candidate_id": parent_trial.get("candidate_id"),
            "trial_id": parent_trial.get("trial_id"),
            "coarse_method": f"{parent_trial.get('coarse_method')}+{refine_config.get('method', 'gicp')}",
            "feature_subset": parent_trial.get("feature_subset"),
            "voxel_size": parent_trial.get("voxel_size"),
            "distance_threshold_factor": parent_trial.get("distance_threshold_factor"),
            "status": "success",
            "algorithm_fitness": float(result.fitness),
            "algorithm_inlier_rmse": float(result.inlier_rmse),
            "algorithm_correspondence_set_size": correspondence_count(result),
            "transformation": require_source_to_target_matrix(result.transformation),
            "time": time.perf_counter() - started,
        }
        return refined
    except Exception as exc:
        return {
            "candidate_id": f"refined_{parent_trial.get('candidate_id')}",
            "parent_candidate_id": parent_trial.get("candidate_id"),
            "status": "failed",
            "error": str(exc),
            "time": time.perf_counter() - started,
        }


def _affine_refine(source_pcd: Any, target_pcd: Any, coarse_transform: np.ndarray, params: Mapping[str, Any], affine_mode: str) -> tuple[np.ndarray, dict[str, float] | None, str]:
    source_points, target_points = _nearest_inlier_pairs(source_pcd, target_pcd, coarse_transform, params)
    if len(source_points) < 4:
        raise RuntimeError("Not enough inlier correspondences for affine refine.")
    if affine_mode == "uniform_scale":
        return _uniform_scale_affine(source_points, target_points, coarse_transform, params)
    if affine_mode == "horizontal_vertical_scale":
        return _horizontal_vertical_affine(source_points, target_points, coarse_transform, params)
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
    max_dist = float(params["voxel_size"]) * float(params.get("affine_pair_distance_factor", params.get("inlier_distance_factor", 2.0)))
    src_pairs: list[np.ndarray] = []
    tgt_pairs: list[np.ndarray] = []
    distances: list[float] = []
    for source_point, transformed_point in zip(source_points, transformed):
        count, idx, squared = tree.search_knn_vector_3d(transformed_point, 1)
        distance = float(np.sqrt(squared[0])) if count else float("inf")
        if count and distance <= max_dist:
            src_pairs.append(source_point)
            tgt_pairs.append(target_points[idx[0]])
            distances.append(distance)
    if not src_pairs:
        return np.asarray(src_pairs, dtype=float), np.asarray(tgt_pairs, dtype=float)
    order = np.argsort(np.asarray(distances, dtype=float))
    keep_ratio = min(1.0, max(0.05, float(params.get("affine_trimmed_ratio", 0.7))))
    keep_count = max(4, int(len(order) * keep_ratio))
    keep = order[:keep_count]
    return np.asarray(src_pairs, dtype=float)[keep], np.asarray(tgt_pairs, dtype=float)[keep]


def _uniform_scale_affine(source_points: np.ndarray, target_points: np.ndarray, coarse_transform: np.ndarray, params: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, float], str]:
    scales = np.linspace(float(params["scale_min"]), float(params["scale_max"]), int(params["scale_steps"]))
    best_transform, best_scale, _ = _grid_scaled_transform(
        source_points,
        target_points,
        coarse_transform,
        ({"sx": float(scale), "sy": float(scale), "sz": float(scale)} for scale in scales),
        params,
    )
    return best_transform, best_scale, "uniform_scale_affine"


def _horizontal_vertical_affine(source_points: np.ndarray, target_points: np.ndarray, coarse_transform: np.ndarray, params: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, float], str]:
    h_min = float(params.get("horizontal_scale_min") or params["scale_min"])
    h_max = float(params.get("horizontal_scale_max") or params["scale_max"])
    h_steps = int(params.get("horizontal_scale_steps") or params["scale_steps"])
    v_min = float(params.get("vertical_scale_min") or params["scale_min"])
    v_max = float(params.get("vertical_scale_max") or params["scale_max"])
    v_steps = int(params.get("vertical_scale_steps") or params["scale_steps"])
    horizontal_scales = np.linspace(h_min, h_max, h_steps)
    vertical_scales = np.linspace(v_min, v_max, v_steps)
    best_transform, best_scale, _ = _grid_scaled_transform(
        source_points,
        target_points,
        coarse_transform,
        ({"sx": float(h), "sy": float(v), "sz": float(h)} for h, v in itertools.product(horizontal_scales, vertical_scales)),
        params,
    )
    return best_transform, best_scale, "horizontal_vertical_scale_affine"


def _constrained_affine(source_points: np.ndarray, target_points: np.ndarray, coarse_transform: np.ndarray, params: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, float], str]:
    scales = np.linspace(float(params["scale_min"]), float(params["scale_max"]), int(params["scale_steps"]))
    best_transform, best_scale, _ = _grid_scaled_transform(
        source_points,
        target_points,
        coarse_transform,
        ({"sx": float(sx), "sy": float(sy), "sz": float(sz)} for sx, sy, sz in itertools.product(scales, repeat=3)),
        params,
    )
    return best_transform, best_scale, "constrained_affine"


def _grid_scaled_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
    coarse_transform: np.ndarray,
    scale_candidates: Any,
    params: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float], float]:
    rotation = coarse_transform[:3, :3]
    best_error = float("inf")
    best_transform = None
    best_scale = None
    trim_ratio = min(1.0, max(0.05, float(params.get("affine_trimmed_ratio", 0.7))))
    keep_count = max(4, int(len(source_points) * trim_ratio))
    for scale_values in scale_candidates:
        scale = np.diag([scale_values["sx"], scale_values["sy"], scale_values["sz"]])
        rotated_scaled = (rotation @ scale @ source_points.T).T
        translation = np.median(target_points - rotated_scaled, axis=0)
        residual = rotated_scaled + translation - target_points
        distances = np.linalg.norm(residual, axis=1)
        keep = np.argsort(distances)[:keep_count]
        translation = np.mean(target_points[keep] - rotated_scaled[keep], axis=0)
        residual = rotated_scaled[keep] + translation - target_points[keep]
        error = float(np.mean(np.sum(residual * residual, axis=1)))
        if error < best_error:
            best_error = error
            best_transform = np.eye(4)
            best_transform[:3, :3] = rotation @ scale
            best_transform[:3, 3] = translation
            best_scale = dict(scale_values)
    if best_transform is None or best_scale is None:
        raise RuntimeError("Constrained affine grid search failed.")
    return best_transform, best_scale, best_error


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
    refine_time: float,
    refined_trials: list[dict[str, Any]],
    transform_type: str,
    scale_values: dict[str, float] | None,
    affine_error: str | None,
) -> dict[str, Any]:
    payload = {
        "coarse_method": params.get("coarse_method"),
        "coarse_methods": params.get("coarse_methods"),
        "coarse_trials": int(params.get("coarse_trials", 3)),
        "candidate_count": len(all_trials),
        "top_k_refine": int(params.get("top_k_refine", 3)),
        "best_trial": best_trial.get("trial_id") if best_trial else None,
        "best_candidate_id": best_trial.get("candidate_id") if best_trial else None,
        "best_score": best_score,
        "all_trials": all_trials,
        "refined_trials": refined_trials,
        "affine_mode": params.get("affine_mode"),
        "transform_type": transform_type,
        "scale_values": scale_values,
        "affine_error": affine_error,
        "coarse_time": coarse_time,
        "refine_time": refine_time,
        "affine_time": affine_time,
    }
    if affine_error:
        payload["affine_failed"] = True
    return payload

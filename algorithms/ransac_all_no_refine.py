from __future__ import annotations

from typing import Any, Mapping

from algorithms import multi_comb
from algorithms.base import RegistrationResult
from algorithms.common import merged_params


METHOD_NAME = "ransac_all_no_refine"


BASE_SCORE_WEIGHTS = {
    "w_fitness": 1.0,
    "w_overlap": 0.5,
    "w_rmse": 0.5,
    "w_trimmed": 0.5,
    "w_translation": 0.0,
    "w_coverage": 1.5,
    "w_degeneracy": 1.2,
    "w_motion_prior": 0.0,
}


DEFAULT_PARAMS = {
    "coarse_methods": ["ransac"],
    "coarse_trials": 5,
    "voxel_sizes": [0.8],
    "distance_threshold_factors": [2.25],
    "feature_subsets": ["all"],
    "top_k_refine": 1,
    "rigid_refine": {"enabled": False},
    "affine_mode": "none",
    "score_weights": BASE_SCORE_WEIGHTS,
}


def run(prepared_source: Any, prepared_target: Any, params: Mapping[str, Any] | None = None) -> RegistrationResult:
    used_params = _merge_params(DEFAULT_PARAMS, params)
    result = multi_comb.run(prepared_source, prepared_target, used_params)
    algorithm_metrics = dict(result.algorithm_metrics)
    algorithm_metrics.update(
        {
            "base_algorithm": multi_comb.METHOD_NAME,
            "selection_policy": "best_score_within_ransac_all_distance_2p25_no_refine",
            "extracted_from": "pure_point_candidate_search_20260604_125056:mixed_no_refine_candidate_generator:candidate_35",
            "source_candidate_signature": {
                "coarse_method": "ransac",
                "feature_subset": "all",
                "voxel_size": 0.8,
                "distance_threshold_factor": 2.25,
                "trial_id": 5,
            },
        }
    )
    return RegistrationResult(
        method=METHOD_NAME,
        status=result.status,
        transformation=result.transformation,
        runtime_sec=result.runtime_sec,
        fitness=result.fitness,
        inlier_rmse=result.inlier_rmse,
        correspondence_set_size=result.correspondence_set_size,
        params=used_params,
        error=result.error,
        metadata=dict(result.metadata),
        algorithm_metrics=algorithm_metrics,
    )


def _merge_params(defaults: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = merged_params(defaults, overrides)
    for key in ("score_weights", "filtering", "coverage", "motion_prior", "rigid_refine"):
        if key in defaults or (overrides and key in overrides):
            merged[key] = dict(defaults.get(key, {})) | dict((overrides or {}).get(key, {}))
    return merged

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from testbench.specified.multi_comb.pair_batch_parameter_test import _write_summary
from utils import io, reporting
from workflows.pairwise import run_pairwise_task


EXPERIMENT_NAME = "pure_point_candidate_search"

SOURCE = {"id": "incremental_16", "path": "data/raw/16/incremental.ply"}
TARGET = {"id": "world_14", "path": "data/raw/14/world.ply"}


BASE_SCORE_WEIGHTS = {
    "w_fitness": 1.0,
    "w_overlap": 0.5,
    "w_rmse": 0.5,
    "w_trimmed": 0.5,
    "w_translation": 0.0,
    "w_coverage": 0.8,
    "w_degeneracy": 0.3,
    "w_motion_prior": 0.0,
}


PARAMETER_SETS = [
    (
        "structure_non_floor_ransac",
        {
            "coarse_methods": ["ransac"],
            "coarse_trials": 8,
            "distance_threshold_factors": [1.5, 2.0, 2.5],
            "feature_subsets": ["non_floor"],
            "top_k_refine": 6,
            "score_weights": BASE_SCORE_WEIGHTS | {"w_coverage": 1.2, "w_degeneracy": 0.8},
            "affine_mode": "none",
        },
    ),
    (
        "near_mid_structure_ransac",
        {
            "coarse_methods": ["ransac"],
            "coarse_trials": 8,
            "distance_threshold_factors": [1.5, 2.0, 2.5],
            "feature_subsets": ["near_mid_only"],
            "top_k_refine": 6,
            "filtering": {"near_mid_radius": 7.5},
            "score_weights": BASE_SCORE_WEIGHTS | {"w_coverage": 1.0, "w_trimmed": 0.8},
            "affine_mode": "none",
        },
    ),
    (
        "mixed_all_non_floor_coverage",
        {
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 5,
            "distance_threshold_factors": [1.25, 1.75, 2.25, 2.75],
            "feature_subsets": ["all", "non_floor"],
            "top_k_refine": 8,
            "score_weights": BASE_SCORE_WEIGHTS | {"w_coverage": 1.8, "w_degeneracy": 1.0, "w_trimmed": 0.8},
            "affine_mode": "none",
        },
    ),
    (
        "mixed_no_refine_candidate_generator",
        {
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 5,
            "distance_threshold_factors": [1.25, 1.75, 2.25, 2.75],
            "feature_subsets": ["all", "non_floor", "near_mid_only"],
            "top_k_refine": 8,
            "rigid_refine": {"enabled": False},
            "score_weights": BASE_SCORE_WEIGHTS | {"w_coverage": 1.5, "w_degeneracy": 1.2},
            "affine_mode": "none",
        },
    ),
    (
        "mixed_no_refine_uniform_scale",
        {
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 5,
            "distance_threshold_factors": [1.25, 1.75, 2.25, 2.75],
            "feature_subsets": ["all", "non_floor", "near_mid_only"],
            "top_k_refine": 8,
            "rigid_refine": {"enabled": False},
            "score_weights": BASE_SCORE_WEIGHTS | {"w_coverage": 1.5, "w_degeneracy": 1.2},
            "affine_mode": "uniform_scale",
            "scale_min": 0.45,
            "scale_max": 1.75,
            "scale_steps": 27,
            "affine_pair_distance_factor": 10.0,
            "affine_trimmed_ratio": 0.6,
        },
    ),
    (
        "mixed_no_refine_horizontal_vertical_scale",
        {
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 5,
            "distance_threshold_factors": [1.25, 1.75, 2.25, 2.75],
            "feature_subsets": ["all", "non_floor", "near_mid_only"],
            "top_k_refine": 8,
            "rigid_refine": {"enabled": False},
            "score_weights": BASE_SCORE_WEIGHTS | {"w_coverage": 1.5, "w_degeneracy": 1.2},
            "affine_mode": "horizontal_vertical_scale",
            "horizontal_scale_min": 0.45,
            "horizontal_scale_max": 1.75,
            "horizontal_scale_steps": 27,
            "vertical_scale_min": 0.45,
            "vertical_scale_max": 1.75,
            "vertical_scale_steps": 27,
            "affine_pair_distance_factor": 10.0,
            "affine_trimmed_ratio": 0.6,
        },
    ),
    (
        "mixed_no_refine_constrained_scale",
        {
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 5,
            "distance_threshold_factors": [1.25, 1.75, 2.25, 2.75],
            "feature_subsets": ["all", "non_floor", "near_mid_only"],
            "top_k_refine": 8,
            "rigid_refine": {"enabled": False},
            "score_weights": BASE_SCORE_WEIGHTS | {"w_coverage": 1.5, "w_degeneracy": 1.2},
            "affine_mode": "constrained",
            "scale_min": 0.55,
            "scale_max": 1.55,
            "scale_steps": 11,
            "affine_pair_distance_factor": 10.0,
            "affine_trimmed_ratio": 0.6,
        },
    ),
    (
        "fine_voxel_non_floor",
        {
            "voxel_sizes": [0.55, 0.7],
            "coarse_methods": ["ransac"],
            "coarse_trials": 5,
            "distance_threshold_factors": [1.5, 2.0],
            "feature_subsets": ["non_floor"],
            "top_k_refine": 6,
            "score_weights": BASE_SCORE_WEIGHTS | {"w_trimmed": 1.0, "w_degeneracy": 0.8},
            "affine_mode": "none",
        },
    ),
    (
        "coarse_voxel_mixed",
        {
            "voxel_sizes": [0.9, 1.1],
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 4,
            "distance_threshold_factors": [1.5, 2.0, 2.5],
            "feature_subsets": ["all", "non_floor"],
            "top_k_refine": 6,
            "score_weights": BASE_SCORE_WEIGHTS | {"w_coverage": 1.4, "w_degeneracy": 0.9},
            "affine_mode": "none",
        },
    ),
    (
        "constrained_affine_structure",
        {
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 5,
            "distance_threshold_factors": [1.5, 2.0, 2.5],
            "feature_subsets": ["non_floor", "near_mid_only"],
            "top_k_refine": 6,
            "score_weights": BASE_SCORE_WEIGHTS | {"w_coverage": 1.3, "w_degeneracy": 1.0},
            "affine_mode": "constrained",
            "scale_min": 0.8,
            "scale_max": 1.2,
            "scale_steps": 9,
        },
    ),
    (
        "strict_symmetric_proxy",
        {
            "coarse_methods": ["ransac"],
            "coarse_trials": 10,
            "distance_threshold_factors": [1.0, 1.25, 1.5],
            "feature_subsets": ["all", "non_floor", "near_mid_only"],
            "top_k_refine": 8,
            "score_weights": BASE_SCORE_WEIGHTS | {"w_fitness": 0.7, "w_rmse": 1.0, "w_trimmed": 1.5, "w_coverage": 1.3, "w_degeneracy": 1.2},
            "affine_mode": "none",
        },
    ),
]


def run_experiment(config: Mapping[str, Any], max_visualizations: int) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    _validate_paths()
    run_dir = reporting.make_run_dir(Path("results") / "specified" / "multi_comb" / EXPERIMENT_NAME, EXPERIMENT_NAME)
    records = []
    for variant_name, params in PARAMETER_SETS:
        records.append(_run_variant(config, run_dir, variant_name, params))
    reporting.write_metrics(run_dir, records)
    diverse = _select_diverse_records(records, max_visualizations=max_visualizations)
    _write_summary(run_dir / "summary.md", records)
    _write_candidate_report(run_dir / "report" / "candidate_visualization_commands.md", diverse)
    _write_candidate_report(run_dir / "candidate_visualization_commands.md", diverse)
    _write_diverse_json(run_dir / "diverse_candidates.json", diverse)
    return run_dir, records, diverse


def _validate_paths() -> None:
    missing = [path for path in (SOURCE["path"], TARGET["path"]) if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing point cloud path(s): " + ", ".join(missing))


def _run_variant(config: Mapping[str, Any], run_dir: Path, variant_name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    pair_task = {
        "task": {"id": f"{EXPERIMENT_NAME}__{variant_name}", "type": "pairwise"},
        "source": {"id": f"{SOURCE['id']}__{variant_name}", "path": SOURCE["path"]},
        "target": TARGET,
        "algorithm": {"name": "multi_comb", "params": _merged_algorithm_params(config, params)},
        "output": {"category": "specified", "experiment": EXPERIMENT_NAME},
    }
    record = run_pairwise_task(pair_task, config, output_dir=run_dir, write_report=False)
    record.update({"experiment_name": EXPERIMENT_NAME, "pair_id": "incremental_16_to_world_14", "variant_name": variant_name})
    return record


def _merged_algorithm_params(config: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(config.get("algorithm", {}).get("params", {}))
    merged.update(dict(params))
    return merged


def _select_diverse_records(records: list[dict[str, Any]], max_visualizations: int) -> list[dict[str, Any]]:
    valid = [record for record in records if record.get("status") == "success" and record.get("matrix_path")]
    ranked = sorted(valid, key=_rank_key, reverse=True)
    selected: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    for record in ranked:
        matrix = io.read_matrix(record["matrix_path"])
        if all(_is_diverse(matrix, other, min_rotation_deg=8.0, min_translation=0.75) for other in matrices):
            selected.append(record)
            matrices.append(matrix)
        if len(selected) >= max_visualizations:
            return selected
    for record in ranked:
        if record in selected:
            continue
        matrix = io.read_matrix(record["matrix_path"])
        if all(_is_diverse(matrix, other, min_rotation_deg=4.0, min_translation=0.4) for other in matrices):
            selected.append(record)
            matrices.append(matrix)
        if len(selected) >= max_visualizations:
            break
    return selected


def _rank_key(record: Mapping[str, Any]) -> float:
    fitness = _float(record.get("eval_fitness"), 0.0)
    trimmed = _float(record.get("eval_trimmed_mean_nn_dist"), 10.0)
    rmse = _float(record.get("eval_inlier_rmse"), 10.0)
    det_penalty = abs(_float(record.get("eval_det_R"), 1.0) - 1.0)
    affine_bonus = 0.05 if record.get("algorithm_transform_type") == "constrained_affine" else 0.0
    return fitness - 0.35 * trimmed - 0.15 * rmse - 0.2 * det_penalty + affine_bonus


def _is_diverse(matrix: np.ndarray, other: np.ndarray, min_rotation_deg: float, min_translation: float) -> bool:
    translation_diff = float(np.linalg.norm(matrix[:3, 3] - other[:3, 3]))
    rotation_diff = _rotation_angle_deg(matrix[:3, :3] @ other[:3, :3].T)
    return translation_diff >= min_translation or rotation_diff >= min_rotation_deg


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) / 2.0
    value = min(1.0, max(-1.0, value))
    return float(np.degrees(np.arccos(value)))


def _write_candidate_report(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Diverse Candidate Visualization Commands", ""]
    for idx, record in enumerate(records, start=1):
        lines.extend(
            [
                f"## Candidate {idx}: {record.get('variant_name')}",
                "",
                f"- transform_type: `{record.get('algorithm_transform_type')}`",
                f"- affine_mode: `{record.get('algorithm_affine_mode')}`",
                f"- scale_values: `{record.get('algorithm_scale_values')}`",
                f"- eval_fitness: `{_fmt(record.get('eval_fitness'))}`",
                f"- eval_inlier_rmse: `{_fmt(record.get('eval_inlier_rmse'))}`",
                f"- eval_trimmed_mean_nn_dist: `{_fmt(record.get('eval_trimmed_mean_nn_dist'))}`",
                f"- algorithm_best_score: `{_fmt(record.get('algorithm_best_score'))}`",
                "",
                "```bat",
                ".env\\python.exe scripts\\visualize_overlay.py --source \"{source}\" --target \"{target}\" --matrix \"{matrix}\"".format(
                    source=record.get("source_path"),
                    target=record.get("target_path"),
                    matrix=record.get("matrix_path"),
                ),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_diverse_json(path: Path, records: list[dict[str, Any]]) -> None:
    selected = [
        {
            "variant_name": record.get("variant_name"),
            "matrix_path": record.get("matrix_path"),
            "transform_type": record.get("algorithm_transform_type"),
            "affine_mode": record.get("algorithm_affine_mode"),
            "scale_values": record.get("algorithm_scale_values"),
            "eval_fitness": record.get("eval_fitness"),
            "eval_inlier_rmse": record.get("eval_inlier_rmse"),
            "eval_trimmed_mean_nn_dist": record.get("eval_trimmed_mean_nn_dist"),
            "algorithm_best_score": record.get("algorithm_best_score"),
        }
        for record in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")


def _float(value: Any, default: float) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Search diverse pure-point-cloud multi_comb candidates.")
    parser.add_argument("--config", default="configs/multi_comb.yaml")
    parser.add_argument("--max-visualizations", type=int, default=6)
    args = parser.parse_args()
    run_dir, records, diverse = run_experiment(io.read_config(args.config), args.max_visualizations)
    print(f"wrote {len(records)} records to {run_dir}")
    print(f"selected {len(diverse)} diverse visualization candidates")
    return 0 if all(record.get("status") == "success" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

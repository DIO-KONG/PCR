from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from testbench.specified.multi_comb.pair_batch_parameter_test import _write_summary, _write_visualization_commands
from utils import io, reporting
from workflows.pairwise import run_pairwise_task


EXPERIMENT_NAME = "incremental16_to_world14_parameter_sweep"

SOURCE = {"id": "incremental_16", "path": "data/raw/16/incremental.ply"}
TARGET = {"id": "world_14", "path": "data/raw/14/world.ply"}


PARAMETER_SETS = [
    (
        "light_ransac_all",
        {
            "coarse_methods": ["ransac"],
            "coarse_trials": 2,
            "distance_threshold_factors": [2.0],
            "feature_subsets": ["all"],
            "top_k_refine": 2,
            "affine_mode": "none",
        },
    ),
    (
        "light_ransac_non_floor",
        {
            "coarse_methods": ["ransac"],
            "coarse_trials": 2,
            "distance_threshold_factors": [2.0],
            "feature_subsets": ["non_floor"],
            "top_k_refine": 2,
            "affine_mode": "none",
        },
    ),
    (
        "light_ransac_near_mid",
        {
            "coarse_methods": ["ransac"],
            "coarse_trials": 2,
            "distance_threshold_factors": [2.0],
            "feature_subsets": ["near_mid_only"],
            "top_k_refine": 2,
            "affine_mode": "none",
        },
    ),
    (
        "fgr_all",
        {
            "coarse_methods": ["fgr"],
            "coarse_trials": 1,
            "distance_threshold_factors": [1.5, 2.0, 2.5],
            "feature_subsets": ["all"],
            "top_k_refine": 2,
            "affine_mode": "none",
        },
    ),
    (
        "fgr_non_floor",
        {
            "coarse_methods": ["fgr"],
            "coarse_trials": 1,
            "distance_threshold_factors": [1.5, 2.0, 2.5],
            "feature_subsets": ["non_floor"],
            "top_k_refine": 2,
            "affine_mode": "none",
        },
    ),
    (
        "ransac_distance_sweep",
        {
            "coarse_methods": ["ransac"],
            "coarse_trials": 3,
            "distance_threshold_factors": [1.5, 2.0, 2.5],
            "feature_subsets": ["all"],
            "top_k_refine": 3,
            "affine_mode": "none",
        },
    ),
    (
        "ransac_subset_sweep",
        {
            "coarse_methods": ["ransac"],
            "coarse_trials": 3,
            "distance_threshold_factors": [2.0],
            "feature_subsets": ["all", "non_floor", "near_mid_only"],
            "top_k_refine": 3,
            "affine_mode": "none",
        },
    ),
    (
        "mixed_robust_top3",
        {
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 2,
            "distance_threshold_factors": [1.5, 2.0, 2.5],
            "feature_subsets": ["all", "non_floor", "near_mid_only"],
            "top_k_refine": 3,
            "affine_mode": "none",
        },
    ),
    (
        "mixed_robust_top5",
        {
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 2,
            "distance_threshold_factors": [1.5, 2.0, 2.5],
            "feature_subsets": ["all", "non_floor", "near_mid_only"],
            "top_k_refine": 5,
            "affine_mode": "none",
        },
    ),
    (
        "mixed_no_refine",
        {
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 2,
            "distance_threshold_factors": [1.5, 2.0, 2.5],
            "feature_subsets": ["all", "non_floor", "near_mid_only"],
            "top_k_refine": 3,
            "rigid_refine": {"enabled": False},
            "affine_mode": "none",
        },
    ),
    (
        "voxel_0p6_mixed",
        {
            "voxel_sizes": [0.6],
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 2,
            "distance_threshold_factors": [2.0],
            "feature_subsets": ["all", "non_floor"],
            "top_k_refine": 3,
            "affine_mode": "none",
        },
    ),
    (
        "voxel_1p0_mixed",
        {
            "voxel_sizes": [1.0],
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 2,
            "distance_threshold_factors": [2.0],
            "feature_subsets": ["all", "non_floor"],
            "top_k_refine": 3,
            "affine_mode": "none",
        },
    ),
    (
        "mixed_constrained_affine",
        {
            "coarse_methods": ["ransac", "fgr"],
            "coarse_trials": 2,
            "distance_threshold_factors": [1.5, 2.0, 2.5],
            "feature_subsets": ["all", "non_floor", "near_mid_only"],
            "top_k_refine": 3,
            "affine_mode": "constrained",
            "scale_min": 0.85,
            "scale_max": 1.15,
            "scale_steps": 7,
        },
    ),
]


def run_experiment(config: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    _validate_paths()
    run_dir = reporting.make_run_dir(Path("results") / "specified" / "multi_comb" / EXPERIMENT_NAME, EXPERIMENT_NAME)
    records = []
    for variant_name, params in PARAMETER_SETS:
        records.append(_run_variant(config, run_dir, variant_name, params))
    reporting.write_metrics(run_dir, records)
    _write_summary(run_dir / "summary.md", records)
    _write_summary(run_dir / "report" / "summary.md", records)
    _write_visualization_commands(run_dir / "report" / "visualization_commands.md", records)
    return run_dir, records


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi_comb parameter sweep on incremental_16 -> world_14.")
    parser.add_argument("--config", default="configs/multi_comb.yaml")
    args = parser.parse_args()
    run_dir, records = run_experiment(io.read_config(args.config))
    print(f"wrote {len(records)} records to {run_dir}")
    return 0 if all(record.get("status") == "success" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

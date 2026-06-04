from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from algorithms.common import correspondence_count, require_source_to_target_matrix
from utils import coverage, degeneracy, io, metrics, point_filters, preprocessing, reporting, visualization


EXPERIMENT_NAME = "ransac_all_candidate_bank"
SOURCE = {"id": "incremental_16", "path": "data/raw/16/incremental.ply"}
TARGET = {"id": "world_14", "path": "data/raw/14/world.ply"}
REFERENCE_MATRIX = (
    "results/specified/multi_comb/pure_point_candidate_search/"
    "pure_point_candidate_search_20260604_125056/matrix/"
    "multi_comb__incremental_16__mixed_no_refine_candidate_generator__to__world_14.txt"
)


SCORE_WEIGHTS = {
    "w_fitness": 1.0,
    "w_overlap": 0.5,
    "w_rmse": 0.5,
    "w_trimmed": 0.5,
    "w_translation": 0.0,
    "w_coverage": 1.5,
    "w_degeneracy": 1.2,
}


def run_experiment(config: Mapping[str, Any], args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    import open3d as o3d

    run_dir = reporting.make_run_dir(Path("results") / "specified" / "multi_comb" / EXPERIMENT_NAME, EXPERIMENT_NAME)
    prep_config = dict(config.get("preprocessing", {}))
    source = preprocessing.prepare_point_cloud_from_path_with_cache(SOURCE["path"], prep_config)
    target = preprocessing.prepare_point_cloud_from_path_with_cache(TARGET["path"], prep_config)
    source_raw = source["raw_pcd"]
    target_raw = target["raw_pcd"]
    source_features, target_features = _prepare_all_features(o3d, source, target, args.voxel_size)
    reference = io.read_matrix(args.reference_matrix)

    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for candidate_id in range(1, args.trials + 1):
        record = _run_candidate(
            o3d,
            source_features,
            target_features,
            source_raw,
            target_raw,
            reference,
            args,
            candidate_id,
        )
        record["elapsed_time"] = time.perf_counter() - start
        matrix = record.pop("_matrix", None)
        if matrix is not None:
            matrix_path = run_dir / "matrix" / f"candidate_{candidate_id:04d}.txt"
            io.write_matrix(matrix_path, matrix)
            record["matrix_path"] = str(matrix_path)
        records.append(record)

    _write_records(run_dir, records)
    selected = _select_candidates(records, args.visualize_top)
    _write_selected_artifacts(run_dir, selected, source_raw, target_raw)
    _write_summary(run_dir, records, selected, args)
    _write_visualization_commands(run_dir / "report" / "candidate_visualization_commands.md", selected)
    return run_dir, records


def _prepare_all_features(o3d: Any, source: Mapping[str, Any], target: Mapping[str, Any], voxel_size: float) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_config = {
        "normal_radius_factor": 2.0,
        "fpfh_radius_factor": 5.0,
        "normal_max_nn": 30,
        "fpfh_max_nn": 100,
    }
    return (
        point_filters.prepare_subset_features(o3d, source["pcd"], voxel_size, feature_config),
        point_filters.prepare_subset_features(o3d, target["pcd"], voxel_size, feature_config),
    )


def _run_candidate(
    o3d: Any,
    source_features: Mapping[str, Any],
    target_features: Mapping[str, Any],
    source_raw: Any,
    target_raw: Any,
    reference: np.ndarray,
    args: argparse.Namespace,
    candidate_id: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    registration = o3d.pipelines.registration
    distance_threshold = args.voxel_size * args.distance_threshold_factor
    try:
        result = registration.registration_ransac_based_on_feature_matching(
            source_features["pcd_down"],
            target_features["pcd_down"],
            source_features["fpfh"],
            target_features["fpfh"],
            True,
            distance_threshold,
            registration.TransformationEstimationPointToPoint(False),
            args.ransac_n,
            [
                registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            registration.RANSACConvergenceCriteria(args.max_iteration, args.confidence),
        )
        matrix = require_source_to_target_matrix(result.transformation)
        record = {
            "candidate_id": candidate_id,
            "status": "success",
            "coarse_method": "ransac",
            "feature_subset": "all",
            "voxel_size": args.voxel_size,
            "distance_threshold_factor": args.distance_threshold_factor,
            "algorithm_fitness": float(result.fitness),
            "algorithm_inlier_rmse": float(result.inlier_rmse),
            "algorithm_correspondence_set_size": correspondence_count(result),
            "runtime": time.perf_counter() - started,
            "_matrix": matrix,
        }
        record.update(metrics.evaluate_registration(source_features["pcd_down"], target_features["pcd_down"], matrix, {"trimmed_ratio": 0.9, "overlap_threshold": distance_threshold}))
        src_inliers, tgt_inliers, _ = coverage.inlier_pairs(source_features["pcd_down"], target_features["pcd_down"], matrix, distance_threshold)
        record["coverage_score"] = coverage.coverage_score(tgt_inliers, np.asarray(target_features["pcd_down"].points), {"coverage_grid_size": 1.0})
        record["plane_degeneracy"] = degeneracy.plane_degeneracy(tgt_inliers)
        record["inlier_count"] = int(len(tgt_inliers))
        record["score"] = _score(record)
        record.update(_similarity_to_reference(matrix, reference))
        return record
    except Exception as exc:
        return {
            "candidate_id": candidate_id,
            "status": "failed",
            "coarse_method": "ransac",
            "feature_subset": "all",
            "voxel_size": args.voxel_size,
            "distance_threshold_factor": args.distance_threshold_factor,
            "runtime": time.perf_counter() - started,
            "error": str(exc),
        }


def _score(record: Mapping[str, Any]) -> float:
    score = 0.0
    terms = [
        ("eval_fitness", "w_fitness", 1.0),
        ("eval_overlap_ratio", "w_overlap", 1.0),
        ("eval_inlier_rmse", "w_rmse", -1.0),
        ("eval_trimmed_mean_nn_dist", "w_trimmed", -1.0),
        ("eval_translation_norm", "w_translation", -1.0),
        ("coverage_score", "w_coverage", 1.0),
        ("plane_degeneracy", "w_degeneracy", -1.0),
    ]
    for metric_key, weight_key, sign in terms:
        value = record.get(metric_key)
        if value is not None:
            score += SCORE_WEIGHTS.get(weight_key, 0.0) * sign * float(value)
    return float(score)


def _similarity_to_reference(matrix: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    translation_diff = float(np.linalg.norm(matrix[:3, 3] - reference[:3, 3]))
    rotation_diff = _rotation_angle_deg(matrix[:3, :3] @ reference[:3, :3].T)
    return {
        "reference_frobenius_diff": float(np.linalg.norm(matrix - reference)),
        "reference_translation_diff": translation_diff,
        "reference_rotation_diff_deg": rotation_diff,
        "reference_similarity_score": translation_diff + 0.1 * rotation_diff,
    }


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) / 2.0
    value = min(1.0, max(-1.0, value))
    return float(np.degrees(np.arccos(value)))


def _select_candidates(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    successful = [record for record in records if record.get("status") == "success" and record.get("matrix_path")]
    by_similarity = sorted(successful, key=lambda r: float(r.get("reference_similarity_score", float("inf"))))[:limit]
    by_score = sorted(successful, key=lambda r: float(r.get("score", float("-inf"))), reverse=True)[: max(3, limit // 2)]
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for record in [*by_similarity, *by_score]:
        candidate_id = int(record["candidate_id"])
        if candidate_id not in seen:
            merged.append(record)
            seen.add(candidate_id)
    return merged[:limit]


def _write_selected_artifacts(run_dir: Path, selected: list[dict[str, Any]], source_raw: Any, target_raw: Any) -> None:
    for record in selected:
        matrix = io.read_matrix(record["matrix_path"])
        stem = f"candidate_{int(record['candidate_id']):04d}"
        cloud_path = run_dir / "cloud" / f"{stem}__registered.ply"
        overlay_path = run_dir / "cloud" / f"{stem}__overlay.ply"
        visualization.save_transformed_cloud(source_raw, matrix, cloud_path)
        visualization.save_overlay_cloud(source_raw, target_raw, matrix, overlay_path)
        record["cloud_path"] = str(cloud_path)
        record["overlay_path"] = str(overlay_path)


def _write_records(run_dir: Path, records: list[dict[str, Any]]) -> None:
    (run_dir / "report").mkdir(parents=True, exist_ok=True)
    serializable = [{k: v for k, v in record.items() if not k.startswith("_")} for record in records]
    (run_dir / "metrics.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = sorted({key for record in serializable for key in record.keys()})
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serializable)


def _write_summary(run_dir: Path, records: list[dict[str, Any]], selected: list[dict[str, Any]], args: argparse.Namespace) -> None:
    successful = [record for record in records if record.get("status") == "success"]
    best_similarity = min(successful, key=lambda r: float(r.get("reference_similarity_score", float("inf"))), default=None)
    best_score = max(successful, key=lambda r: float(r.get("score", float("-inf"))), default=None)
    lines = [
        "# RANSAC All Candidate Bank",
        "",
        f"- source: `{SOURCE['path']}`",
        f"- target: `{TARGET['path']}`",
        f"- reference_matrix: `{args.reference_matrix}`",
        f"- trials: `{args.trials}`",
        f"- successful: `{len(successful)}`",
        f"- voxel_size: `{args.voxel_size}`",
        f"- distance_threshold_factor: `{args.distance_threshold_factor}`",
        "",
    ]
    for title, record in [("Best Similar To Reference", best_similarity), ("Best By Internal Score", best_score)]:
        lines.extend(_summary_block(title, record))
    lines.extend(["## Selected Visualization Candidates", "", "| candidate | ref_t_diff | ref_r_diff_deg | score | eval_fitness | trimmed | matrix |", "|---:|---:|---:|---:|---:|---:|---|"])
    for record in selected:
        lines.append(
            "| {candidate_id} | {td:.6f} | {rd:.6f} | {score:.6f} | {fitness:.6f} | {trimmed:.6f} | {matrix} |".format(
                candidate_id=int(record["candidate_id"]),
                td=float(record.get("reference_translation_diff", 0.0)),
                rd=float(record.get("reference_rotation_diff_deg", 0.0)),
                score=float(record.get("score", 0.0)),
                fitness=float(record.get("eval_fitness", 0.0)),
                trimmed=float(record.get("eval_trimmed_mean_nn_dist", 0.0)),
                matrix=record.get("matrix_path"),
            )
        )
    text = "\n".join(lines)
    (run_dir / "summary.md").write_text(text, encoding="utf-8")
    (run_dir / "report" / "summary.md").write_text(text, encoding="utf-8")


def _summary_block(title: str, record: Mapping[str, Any] | None) -> list[str]:
    if record is None:
        return [f"## {title}", "", "- No successful candidate.", ""]
    return [
        f"## {title}",
        "",
        f"- candidate_id: `{record.get('candidate_id')}`",
        f"- reference_translation_diff: `{_fmt(record.get('reference_translation_diff'))}`",
        f"- reference_rotation_diff_deg: `{_fmt(record.get('reference_rotation_diff_deg'))}`",
        f"- reference_frobenius_diff: `{_fmt(record.get('reference_frobenius_diff'))}`",
        f"- score: `{_fmt(record.get('score'))}`",
        f"- eval_fitness: `{_fmt(record.get('eval_fitness'))}`",
        f"- eval_trimmed_mean_nn_dist: `{_fmt(record.get('eval_trimmed_mean_nn_dist'))}`",
        f"- matrix: `{record.get('matrix_path')}`",
        "",
    ]


def _write_visualization_commands(path: Path, records: list[dict[str, Any]]) -> None:
    lines = ["# Candidate Visualization Commands", ""]
    for record in records:
        lines.extend(
            [
                f"## Candidate {int(record['candidate_id']):04d}",
                "",
                f"- reference_translation_diff: `{_fmt(record.get('reference_translation_diff'))}`",
                f"- reference_rotation_diff_deg: `{_fmt(record.get('reference_rotation_diff_deg'))}`",
                f"- score: `{_fmt(record.get('score'))}`",
                "",
                "```bat",
                ".env\\python.exe scripts\\visualize_overlay.py --source \"{source}\" --target \"{target}\" --matrix \"{matrix}\"".format(
                    source=SOURCE["path"],
                    target=TARGET["path"],
                    matrix=record["matrix_path"],
                ),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    global SOURCE, TARGET

    parser = argparse.ArgumentParser(description="Run a RANSAC/all-points candidate bank and compare candidates to a reference matrix.")
    parser.add_argument("--config", default="configs/ransac_all_no_refine.yaml")
    parser.add_argument("--source", default=SOURCE["path"])
    parser.add_argument("--target", default=TARGET["path"])
    parser.add_argument("--source-id", default=SOURCE["id"])
    parser.add_argument("--target-id", default=TARGET["id"])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--visualize-top", type=int, default=12)
    parser.add_argument("--reference-matrix", default=REFERENCE_MATRIX)
    parser.add_argument("--voxel-size", type=float, default=0.8)
    parser.add_argument("--distance-threshold-factor", type=float, default=2.25)
    parser.add_argument("--ransac-n", type=int, default=4)
    parser.add_argument("--max-iteration", type=int, default=100000)
    parser.add_argument("--confidence", type=float, default=0.999)
    args = parser.parse_args()
    SOURCE = {"id": args.source_id, "path": args.source}
    TARGET = {"id": args.target_id, "path": args.target}
    run_dir, records = run_experiment(io.read_config(args.config), args)
    successful = [record for record in records if record.get("status") == "success"]
    best = min(successful, key=lambda r: float(r.get("reference_similarity_score", float("inf"))), default=None)
    print(f"wrote {len(records)} candidates to {run_dir}")
    if best:
        print(
            "best_reference_match candidate_id={candidate_id} translation_diff={td:.6f} rotation_diff_deg={rd:.6f} matrix={matrix}".format(
                candidate_id=best["candidate_id"],
                td=float(best["reference_translation_diff"]),
                rd=float(best["reference_rotation_diff_deg"]),
                matrix=best["matrix_path"],
            )
        )
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())

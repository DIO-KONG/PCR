from __future__ import annotations

import argparse
import copy
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

from utils import io, metrics, preprocessing, reporting, visualization


EXPERIMENT_NAME = "cluster_scale_refine"
SOURCE = {"id": "incremental_16", "path": "data/raw/16/incremental.ply"}
TARGET = {"id": "world_14", "path": "data/raw/14/world.ply"}
DEFAULT_BANK_DIR = "results/specified/multi_comb/ransac_all_candidate_bank/ransac_all_candidate_bank_20260604_133558"
DEFAULT_REFERENCE_MATRIX = (
    "results/specified/multi_comb/pure_point_candidate_search/"
    "pure_point_candidate_search_20260604_125056/matrix/"
    "multi_comb__incremental_16__mixed_no_refine_candidate_generator__to__world_14.txt"
)


def run_experiment(config: Mapping[str, Any], args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    import open3d as o3d

    started = time.perf_counter()
    run_dir = reporting.make_run_dir(Path("results") / "specified" / "multi_comb" / EXPERIMENT_NAME, EXPERIMENT_NAME)
    bank_records = _load_bank_records(Path(args.bank_dir))
    for record in bank_records:
        record["_matrix"] = io.read_matrix(record["matrix_path"])
    clusters = _cluster_by_rotation(bank_records, args.cluster_rotation_deg)
    top_score_ids = {int(record["candidate_id"]) for record in sorted(bank_records, key=lambda r: float(r.get("score", float("-inf"))), reverse=True)[: args.top_score_candidates]}
    selected = _select_for_refine(clusters, top_score_ids, args.top_clusters)

    prepared_source = preprocessing.prepare_point_cloud_from_path_with_cache(SOURCE["path"], dict(config.get("preprocessing", {})))
    prepared_target = preprocessing.prepare_point_cloud_from_path_with_cache(TARGET["path"], dict(config.get("preprocessing", {})))
    source_down = prepared_source["pcd_down"]
    target_down = prepared_target["pcd_down"]
    source_raw = prepared_source["raw_pcd"]
    target_raw = prepared_target["raw_pcd"]
    reference = io.read_matrix(args.reference_matrix) if args.reference_matrix else None

    records: list[dict[str, Any]] = []
    for cluster in clusters:
        for record in cluster["items"]:
            record["cluster_id"] = int(cluster["cluster_id"])
            record["cluster_size"] = len(cluster["items"])
            record["cluster_top_score_count"] = sum(int(item["candidate_id"]) in top_score_ids for item in cluster["items"])

    refine_started = time.perf_counter()
    for candidate in selected:
        refined = _scale_and_refine_candidate(o3d, source_down, target_down, candidate, args)
        if reference is not None:
            refined.update(_similarity_to_reference(refined["_final_matrix"], reference))
            refined["is_reference_family"] = refined["reference_rotation_diff_deg"] <= args.correct_rotation_deg
        records.append(refined)
    refine_time = time.perf_counter() - refine_started

    cluster_summaries = _summarize_clusters(clusters, records, top_score_ids)
    ranked_clusters = sorted(
        cluster_summaries,
        key=lambda c: (
            int(c["cluster_top_score_count"]),
            int(c["cluster_size"]),
            _float(c.get("best_refined_rank_score"), float("-inf")),
        ),
        reverse=True,
    )
    top_ranked = ranked_clusters[: args.visualize_clusters]
    _write_outputs(run_dir, records, cluster_summaries, ranked_clusters, top_ranked, source_raw, target_raw, args, time.perf_counter() - started, refine_time)
    return run_dir, records, ranked_clusters


def _load_bank_records(bank_dir: Path) -> list[dict[str, Any]]:
    records = json.loads((bank_dir / "metrics.json").read_text(encoding="utf-8"))
    return [record for record in records if record.get("status") == "success" and record.get("matrix_path")]


def _cluster_by_rotation(records: list[dict[str, Any]], threshold_deg: float) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: float(r.get("score", float("-inf"))), reverse=True):
        placed = False
        for cluster in clusters:
            if _rotation_diff_deg(record["_matrix"], cluster["rep"]["_matrix"]) <= threshold_deg:
                cluster["items"].append(record)
                placed = True
                break
        if not placed:
            clusters.append({"cluster_id": len(clusters) + 1, "rep": record, "items": [record]})
    return sorted(clusters, key=lambda c: len(c["items"]), reverse=True)


def _select_for_refine(clusters: list[dict[str, Any]], top_score_ids: set[int], top_clusters: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for cluster in clusters[:top_clusters]:
        for item in cluster["items"]:
            _append_unique(selected, seen, item)
        top_in_cluster = sorted(cluster["items"], key=lambda r: float(r.get("score", float("-inf"))), reverse=True)[:3]
        for item in top_in_cluster:
            _append_unique(selected, seen, item)
    for cluster in clusters:
        for item in cluster["items"]:
            if int(item["candidate_id"]) in top_score_ids:
                _append_unique(selected, seen, item)
    return selected


def _append_unique(selected: list[dict[str, Any]], seen: set[int], item: dict[str, Any]) -> None:
    candidate_id = int(item["candidate_id"])
    if candidate_id not in seen:
        selected.append(item)
        seen.add(candidate_id)


def _scale_and_refine_candidate(o3d: Any, source_down: Any, target_down: Any, candidate: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    coarse = np.asarray(candidate["_matrix"], dtype=float)
    scale_started = time.perf_counter()
    scaled_matrix, scale_values, scale_error = _horizontal_vertical_scale(source_down, target_down, coarse, args)
    scale_time = time.perf_counter() - scale_started
    gicp_started = time.perf_counter()
    final_matrix, gicp_status, gicp_reason, drift = _bounded_gicp(o3d, source_down, target_down, coarse, scaled_matrix, scale_values, args)
    gicp_time = time.perf_counter() - gicp_started
    eval_metrics = metrics.evaluate_registration(source_down, target_down, final_matrix, {"trimmed_ratio": 0.9, "overlap_threshold": args.eval_overlap_threshold})
    rank_score = _rank_score(eval_metrics, candidate, gicp_status)
    record = {
        "candidate_id": int(candidate["candidate_id"]),
        "cluster_id": int(candidate["cluster_id"]),
        "cluster_size": int(candidate["cluster_size"]),
        "cluster_top_score_count": int(candidate["cluster_top_score_count"]),
        "coarse_score": candidate.get("score"),
        "coarse_eval_fitness": candidate.get("eval_fitness"),
        "coarse_eval_trimmed_mean_nn_dist": candidate.get("eval_trimmed_mean_nn_dist"),
        "scale_values": scale_values,
        "scale_error": scale_error,
        "gicp_status": gicp_status,
        "gicp_reject_reason": gicp_reason,
        "gicp_rotation_drift_deg": drift["rotation_deg"],
        "gicp_translation_drift": drift["translation"],
        "scale_time": scale_time,
        "gicp_time": gicp_time,
        "candidate_total_time": time.perf_counter() - started,
        "refined_rank_score": rank_score,
        "_coarse_matrix": coarse,
        "_scaled_matrix": scaled_matrix,
        "_final_matrix": final_matrix,
    }
    record.update(eval_metrics)
    return record


def _horizontal_vertical_scale(source_pcd: Any, target_pcd: Any, coarse: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, float], float]:
    source_points = np.asarray(source_pcd.points, dtype=float)
    target_points = np.asarray(target_pcd.points, dtype=float)
    transformed = (coarse[:3, :3] @ source_points.T).T + coarse[:3, 3]
    tree = _target_tree(target_pcd)
    max_dist = float(args.scale_pair_distance)
    src_pairs = []
    tgt_pairs = []
    dists = []
    for source_point, transformed_point in zip(source_points, transformed):
        count, idx, squared = tree.search_knn_vector_3d(transformed_point, 1)
        if count:
            dist = float(np.sqrt(squared[0]))
            if dist <= max_dist:
                src_pairs.append(source_point)
                tgt_pairs.append(target_points[idx[0]])
                dists.append(dist)
    if len(src_pairs) < 8:
        return coarse.copy(), {"sx": 1.0, "sy": 1.0, "sz": 1.0}, float("inf")
    order = np.argsort(np.asarray(dists, dtype=float))
    keep = order[: max(8, int(len(order) * args.scale_pair_trimmed_ratio))]
    src = np.asarray(src_pairs, dtype=float)[keep]
    tgt = np.asarray(tgt_pairs, dtype=float)[keep]
    h_values = np.linspace(args.horizontal_scale_min, args.horizontal_scale_max, args.horizontal_scale_steps)
    v_values = np.linspace(args.vertical_scale_min, args.vertical_scale_max, args.vertical_scale_steps)
    best_matrix = coarse.copy()
    best_scale = {"sx": 1.0, "sy": 1.0, "sz": 1.0}
    best_error = float("inf")
    rotation = coarse[:3, :3]
    local_keep = max(8, int(len(src) * args.scale_residual_trimmed_ratio))
    for h in h_values:
        for v in v_values:
            scale = np.diag([h, v, h])
            scaled_rotated = (rotation @ scale @ src.T).T
            translation = np.median(tgt - scaled_rotated, axis=0)
            residual = np.linalg.norm(scaled_rotated + translation - tgt, axis=1)
            idx = np.argsort(residual)[:local_keep]
            translation = np.mean(tgt[idx] - scaled_rotated[idx], axis=0)
            residual = scaled_rotated[idx] + translation - tgt[idx]
            error = float(np.mean(np.sum(residual * residual, axis=1)))
            if error < best_error:
                best_error = error
                best_matrix = np.eye(4)
                best_matrix[:3, :3] = rotation @ scale
                best_matrix[:3, 3] = translation
                best_scale = {"sx": float(h), "sy": float(v), "sz": float(h)}
    return best_matrix, best_scale, best_error


def _target_tree(target_pcd: Any) -> Any:
    import open3d as o3d

    return o3d.geometry.KDTreeFlann(target_pcd)


def _bounded_gicp(o3d: Any, source_pcd: Any, target_pcd: Any, coarse: np.ndarray, scaled_matrix: np.ndarray, scale_values: Mapping[str, float], args: argparse.Namespace) -> tuple[np.ndarray, str, str | None, dict[str, float]]:
    scale = np.eye(4)
    scale[:3, :3] = np.diag([scale_values["sx"], scale_values["sy"], scale_values["sz"]])
    scaled_source = copy.deepcopy(source_pcd)
    scaled_source.transform(scale)
    init = np.eye(4)
    init[:3, :3] = coarse[:3, :3]
    init[:3, 3] = scaled_matrix[:3, 3]
    try:
        registration = o3d.pipelines.registration
        result = registration.registration_generalized_icp(
            scaled_source,
            target_pcd,
            args.gicp_max_correspondence_distance,
            init,
            registration.TransformationEstimationForGeneralizedICP(),
            registration.ICPConvergenceCriteria(max_iteration=args.gicp_max_iterations),
        )
        refined_scaled_source_to_target = np.asarray(result.transformation, dtype=float)
        rotation_drift = _rotation_angle_deg(refined_scaled_source_to_target[:3, :3] @ coarse[:3, :3].T)
        translation_drift = float(np.linalg.norm(refined_scaled_source_to_target[:3, 3] - init[:3, 3]))
        drift = {"rotation_deg": rotation_drift, "translation": translation_drift}
        if rotation_drift > args.bound_rotation_deg:
            return scaled_matrix, "rejected", "rotation_drift_exceeded", drift
        if translation_drift > args.bound_translation:
            return scaled_matrix, "rejected", "translation_drift_exceeded", drift
        return refined_scaled_source_to_target @ scale, "accepted", None, drift
    except Exception as exc:
        return scaled_matrix, "failed", str(exc), {"rotation_deg": 0.0, "translation": 0.0}


def _rank_score(eval_metrics: Mapping[str, Any], coarse: Mapping[str, Any], gicp_status: str) -> float:
    fitness = float(eval_metrics.get("eval_fitness") or 0.0)
    trimmed = float(eval_metrics.get("eval_trimmed_mean_nn_dist") or 10.0)
    rmse = float(eval_metrics.get("eval_inlier_rmse") or 10.0)
    status_bonus = 0.03 if gicp_status == "accepted" else 0.0
    return fitness - 0.35 * trimmed - 0.15 * rmse + 0.08 * float(coarse.get("score") or 0.0) + status_bonus


def _summarize_clusters(clusters: list[dict[str, Any]], records: list[dict[str, Any]], top_score_ids: set[int]) -> list[dict[str, Any]]:
    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_cluster.setdefault(int(record["cluster_id"]), []).append(record)
    summaries = []
    for cluster in clusters:
        cluster_id = int(cluster["cluster_id"])
        refined = by_cluster.get(cluster_id, [])
        best = max(refined, key=lambda r: float(r.get("refined_rank_score", float("-inf"))), default=None)
        summaries.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(cluster["items"]),
                "cluster_top_score_count": sum(int(item["candidate_id"]) in top_score_ids for item in cluster["items"]),
                "candidate_ids": [int(item["candidate_id"]) for item in cluster["items"]],
                "best_candidate_id": best.get("candidate_id") if best else None,
                "best_refined_rank_score": best.get("refined_rank_score") if best else None,
                "best_eval_fitness": best.get("eval_fitness") if best else None,
                "best_eval_trimmed_mean_nn_dist": best.get("eval_trimmed_mean_nn_dist") if best else None,
                "best_is_reference_family": best.get("is_reference_family") if best else None,
            }
        )
    return summaries


def _write_outputs(
    run_dir: Path,
    records: list[dict[str, Any]],
    cluster_summaries: list[dict[str, Any]],
    ranked_clusters: list[dict[str, Any]],
    top_ranked: list[dict[str, Any]],
    source_raw: Any,
    target_raw: Any,
    args: argparse.Namespace,
    total_time: float,
    refine_time: float,
) -> None:
    serializable_records = [{k: v for k, v in record.items() if not k.startswith("_")} for record in records]
    _write_json(run_dir / "metrics.json", serializable_records)
    _write_json(run_dir / "cluster_summary.json", cluster_summaries)
    _write_csv(run_dir / "metrics.csv", serializable_records)
    best_by_cluster = []
    record_by_id = {int(record["candidate_id"]): record for record in records}
    for cluster in top_ranked:
        candidate_id = cluster.get("best_candidate_id")
        if candidate_id is None:
            continue
        record = record_by_id[int(candidate_id)]
        matrix_path = run_dir / "matrix" / f"cluster_{int(cluster['cluster_id']):03d}__candidate_{int(candidate_id):04d}__refined.txt"
        scaled_path = run_dir / "matrix" / f"cluster_{int(cluster['cluster_id']):03d}__candidate_{int(candidate_id):04d}__scaled.txt"
        io.write_matrix(matrix_path, record["_final_matrix"])
        io.write_matrix(scaled_path, record["_scaled_matrix"])
        record["refined_matrix_path"] = str(matrix_path)
        record["scaled_matrix_path"] = str(scaled_path)
        stem = f"cluster_{int(cluster['cluster_id']):03d}__candidate_{int(candidate_id):04d}"
        cloud_path = run_dir / "cloud" / f"{stem}__registered.ply"
        overlay_path = run_dir / "cloud" / f"{stem}__overlay.ply"
        visualization.save_transformed_cloud(source_raw, record["_final_matrix"], cloud_path)
        visualization.save_overlay_cloud(source_raw, target_raw, record["_final_matrix"], overlay_path)
        record["cloud_path"] = str(cloud_path)
        record["overlay_path"] = str(overlay_path)
        best_by_cluster.append(record)
    _write_summary(run_dir, ranked_clusters, best_by_cluster, args, total_time, refine_time)
    _write_visualization_commands(run_dir / "report" / "top_cluster_visualization_commands.md", best_by_cluster)
    _write_json(run_dir / "selected_top_clusters.json", [{k: v for k, v in record.items() if not k.startswith("_")} for record in best_by_cluster])


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(run_dir: Path, ranked_clusters: list[dict[str, Any]], selected: list[dict[str, Any]], args: argparse.Namespace, total_time: float, refine_time: float) -> None:
    lines = [
        "# Cluster Scale Refine",
        "",
        f"- bank_dir: `{args.bank_dir}`",
        f"- top_clusters: `{args.top_clusters}`",
        f"- top_score_candidates: `{args.top_score_candidates}`",
        f"- bound_rotation_deg: `{args.bound_rotation_deg}`",
        f"- bound_translation: `{args.bound_translation}`",
        f"- total_time: `{total_time:.6f}`",
        f"- refine_time: `{refine_time:.6f}`",
        "",
        "## Ranked Clusters",
        "",
        "| rank | cluster | size | top10_count | best_candidate | best_score | best_fitness | best_trimmed | reference_family |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, cluster in enumerate(ranked_clusters[:10], start=1):
        lines.append(
            "| {rank} | {cluster_id} | {size} | {top_count} | {candidate} | {score} | {fitness} | {trimmed} | {ref_family} |".format(
                rank=rank,
                cluster_id=cluster["cluster_id"],
                size=cluster["cluster_size"],
                top_count=cluster["cluster_top_score_count"],
                candidate=cluster.get("best_candidate_id"),
                score=_fmt(cluster.get("best_refined_rank_score")),
                fitness=_fmt(cluster.get("best_eval_fitness")),
                trimmed=_fmt(cluster.get("best_eval_trimmed_mean_nn_dist")),
                ref_family=cluster.get("best_is_reference_family"),
            )
        )
    lines.extend(["", "## Top Cluster Representatives", ""])
    for record in selected:
        lines.extend(
            [
                f"### cluster {record.get('cluster_id')} / candidate {record.get('candidate_id')}",
                "",
                f"- refined_matrix: `{record.get('refined_matrix_path')}`",
                f"- scaled_matrix: `{record.get('scaled_matrix_path')}`",
                f"- gicp_status: `{record.get('gicp_status')}`",
                f"- gicp_rotation_drift_deg: `{_fmt(record.get('gicp_rotation_drift_deg'))}`",
                f"- gicp_translation_drift: `{_fmt(record.get('gicp_translation_drift'))}`",
                f"- scale_values: `{record.get('scale_values')}`",
                f"- eval_fitness: `{_fmt(record.get('eval_fitness'))}`",
                f"- eval_trimmed_mean_nn_dist: `{_fmt(record.get('eval_trimmed_mean_nn_dist'))}`",
                f"- is_reference_family: `{record.get('is_reference_family')}`",
                "",
            ]
        )
    text = "\n".join(lines)
    (run_dir / "summary.md").write_text(text, encoding="utf-8")
    (run_dir / "report" / "summary.md").write_text(text, encoding="utf-8")


def _write_visualization_commands(path: Path, records: list[dict[str, Any]]) -> None:
    lines = ["# Top Cluster Visualization Commands", ""]
    for record in records:
        lines.extend(
            [
                f"## cluster {record.get('cluster_id')} / candidate {record.get('candidate_id')}",
                "",
                f"- matrix: `{record.get('refined_matrix_path')}`",
                f"- gicp_status: `{record.get('gicp_status')}`",
                f"- scale_values: `{record.get('scale_values')}`",
                f"- is_reference_family: `{record.get('is_reference_family')}`",
                "",
                "```bat",
                ".env\\python.exe scripts\\visualize_overlay.py --source \"{source}\" --target \"{target}\" --matrix \"{matrix}\"".format(
                    source=SOURCE["path"],
                    target=TARGET["path"],
                    matrix=record.get("refined_matrix_path"),
                ),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _similarity_to_reference(matrix: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    matrix_rotation = _rotation_component(matrix[:3, :3])
    reference_rotation = _rotation_component(reference[:3, :3])
    return {
        "reference_frobenius_diff": float(np.linalg.norm(matrix - reference)),
        "reference_translation_diff": float(np.linalg.norm(matrix[:3, 3] - reference[:3, 3])),
        "reference_rotation_diff_deg": _rotation_angle_deg(matrix_rotation @ reference_rotation.T),
    }


def _rotation_diff_deg(a: np.ndarray, b: np.ndarray) -> float:
    return _rotation_angle_deg(a[:3, :3] @ b[:3, :3].T)


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (float(np.trace(rotation)) - 1.0) / 2.0
    value = min(1.0, max(-1.0, value))
    return float(np.degrees(np.arccos(value)))


def _rotation_component(matrix3: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix3)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _float(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    parser = argparse.ArgumentParser(description="Scale and bounded-GICP refine selected RANSAC candidate-bank clusters.")
    parser.add_argument("--config", default="configs/ransac_all_no_refine.yaml")
    parser.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    parser.add_argument("--reference-matrix", default=DEFAULT_REFERENCE_MATRIX)
    parser.add_argument("--top-clusters", type=int, default=5)
    parser.add_argument("--top-score-candidates", type=int, default=10)
    parser.add_argument("--visualize-clusters", type=int, default=3)
    parser.add_argument("--cluster-rotation-deg", type=float, default=5.0)
    parser.add_argument("--correct-rotation-deg", type=float, default=5.0)
    parser.add_argument("--horizontal-scale-min", type=float, default=0.7)
    parser.add_argument("--horizontal-scale-max", type=float, default=1.3)
    parser.add_argument("--horizontal-scale-steps", type=int, default=25)
    parser.add_argument("--vertical-scale-min", type=float, default=0.7)
    parser.add_argument("--vertical-scale-max", type=float, default=1.3)
    parser.add_argument("--vertical-scale-steps", type=int, default=25)
    parser.add_argument("--scale-pair-distance", type=float, default=4.0)
    parser.add_argument("--scale-pair-trimmed-ratio", type=float, default=0.7)
    parser.add_argument("--scale-residual-trimmed-ratio", type=float, default=0.7)
    parser.add_argument("--gicp-max-correspondence-distance", type=float, default=1.2)
    parser.add_argument("--gicp-max-iterations", type=int, default=20)
    parser.add_argument("--bound-rotation-deg", type=float, default=5.0)
    parser.add_argument("--bound-translation", type=float, default=1.0)
    parser.add_argument("--eval-overlap-threshold", type=float, default=1.0)
    args = parser.parse_args()
    run_dir, records, ranked_clusters = run_experiment(io.read_config(args.config), args)
    print(f"wrote {len(records)} refined candidates to {run_dir}")
    for idx, cluster in enumerate(ranked_clusters[:3], start=1):
        print(
            "rank={rank} cluster={cluster} size={size} top10_count={top_count} best_candidate={candidate} reference_family={ref_family}".format(
                rank=idx,
                cluster=cluster["cluster_id"],
                size=cluster["cluster_size"],
                top_count=cluster["cluster_top_score_count"],
                candidate=cluster.get("best_candidate_id"),
                ref_family=cluster.get("best_is_reference_family"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

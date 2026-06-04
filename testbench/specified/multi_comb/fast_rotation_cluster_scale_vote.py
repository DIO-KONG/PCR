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

from algorithms.common import correspondence_count, require_source_to_target_matrix
from utils import io, metrics, point_filters, preprocessing, reporting, visualization


EXPERIMENT_NAME = "fast_rotation_cluster_scale_vote"
SOURCE = {"id": "incremental_16", "path": "data/raw/16/incremental.ply"}
TARGET = {"id": "world_14", "path": "data/raw/14/world.ply"}
REFERENCE_MATRIX = (
    "results/specified/multi_comb/pure_point_candidate_search/"
    "pure_point_candidate_search_20260604_125056/matrix/"
    "multi_comb__incremental_16__mixed_no_refine_candidate_generator__to__world_14.txt"
)


def run_experiment(config: Mapping[str, Any], args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    import open3d as o3d

    started = time.perf_counter()
    run_dir = reporting.make_run_dir(Path("results") / "specified" / "multi_comb" / EXPERIMENT_NAME, EXPERIMENT_NAME)
    prep_started = time.perf_counter()
    prepared_source = preprocessing.prepare_point_cloud_from_path_with_cache(SOURCE["path"], dict(config.get("preprocessing", {})))
    prepared_target = preprocessing.prepare_point_cloud_from_path_with_cache(TARGET["path"], dict(config.get("preprocessing", {})))
    source_raw = prepared_source["raw_pcd"]
    target_raw = prepared_target["raw_pcd"]
    source_features, target_features = _prepare_features(o3d, prepared_source, prepared_target, args.voxel_size)
    prep_time = time.perf_counter() - prep_started

    coarse_started = time.perf_counter()
    candidates = _generate_candidates(o3d, source_features, target_features, args)
    coarse_time = time.perf_counter() - coarse_started
    clusters = _cluster_by_rotation(candidates, args.cluster_rotation_deg)
    top_score_ids = {int(item["candidate_id"]) for item in sorted(candidates, key=lambda r: float(r.get("coarse_score", float("-inf"))), reverse=True)[: args.top_score_candidates]}
    selected = _select_for_refine(clusters, top_score_ids, args.top_clusters, args.max_refine_candidates)
    for cluster in clusters:
        for item in cluster["items"]:
            item["cluster_id"] = int(cluster["cluster_id"])
            item["cluster_size"] = len(cluster["items"])
            item["cluster_top_score_count"] = sum(int(member["candidate_id"]) in top_score_ids for member in cluster["items"])

    refine_started = time.perf_counter()
    refined_records = [_scale_vote_refine(o3d, source_features["pcd_down"], target_features["pcd_down"], item, args) for item in selected]
    refine_time = time.perf_counter() - refine_started
    reference = io.read_matrix(args.reference_matrix) if args.reference_matrix else None
    if reference is not None:
        for record in refined_records:
            record.update(_similarity_to_reference(record["_final_matrix"], reference))
            record["is_reference_family"] = record["reference_rotation_diff_deg"] <= args.correct_rotation_deg

    cluster_summaries = _summarize_clusters(clusters, refined_records, top_score_ids)
    ranked_clusters = sorted(
        cluster_summaries,
        key=lambda c: (
            _float(c.get("best_refined_rank_score"), float("-inf")),
            int(c["cluster_top_score_count"]),
            int(c["cluster_size"]),
        ),
        reverse=True,
    )
    total_time = time.perf_counter() - started
    _write_outputs(
        run_dir,
        candidates,
        refined_records,
        ranked_clusters,
        source_raw,
        target_raw,
        args,
        {
            "preprocess_time": prep_time,
            "coarse_time": coarse_time,
            "refine_time": refine_time,
            "total_time": total_time,
        },
    )
    return run_dir, refined_records, ranked_clusters


def _prepare_features(o3d: Any, source: Mapping[str, Any], target: Mapping[str, Any], voxel_size: float) -> tuple[dict[str, Any], dict[str, Any]]:
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


def _generate_candidates(o3d: Any, source_features: Mapping[str, Any], target_features: Mapping[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.candidate_mode == "topview":
        return _generate_topview_candidates(source_features["pcd_down"], target_features["pcd_down"], args)
    return _generate_ransac_candidates(o3d, source_features, target_features, args)


def _generate_ransac_candidates(o3d: Any, source_features: Mapping[str, Any], target_features: Mapping[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    registration = o3d.pipelines.registration
    distance_threshold = args.voxel_size * args.distance_threshold_factor
    candidates = []
    for candidate_id in range(1, args.trials + 1):
        started = time.perf_counter()
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
            corr_count = correspondence_count(result)
            fitness = float(result.fitness)
            rmse = float(result.inlier_rmse)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "status": "success",
                    "algorithm_fitness": fitness,
                    "algorithm_inlier_rmse": rmse,
                    "algorithm_correspondence_set_size": corr_count,
                    "coarse_score": fitness - 0.35 * rmse + 0.0002 * float(corr_count or 0),
                    "runtime": time.perf_counter() - started,
                    "_matrix": matrix,
                }
            )
        except Exception as exc:
            candidates.append({"candidate_id": candidate_id, "status": "failed", "error": str(exc), "runtime": time.perf_counter() - started})
    return [item for item in candidates if item.get("status") == "success"]


def _generate_topview_candidates(source_down: Any, target_down: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    source_points = np.asarray(source_down.points, dtype=float)
    target_points = np.asarray(target_down.points, dtype=float)
    if len(source_points) == 0 or len(target_points) == 0:
        return []

    resolution = float(args.topview_grid_resolution)
    pad = float(args.topview_padding)
    mins = np.minimum(source_points[:, [0, 2]].min(axis=0), target_points[:, [0, 2]].min(axis=0)) - pad
    maxs = np.maximum(source_points[:, [0, 2]].max(axis=0), target_points[:, [0, 2]].max(axis=0)) + pad
    shape = np.ceil((maxs - mins) / resolution).astype(int) + 1
    target_hist = _topview_histogram(target_points, mins, shape, resolution)
    target_fft = np.fft.rfftn(target_hist)
    source_median_y = float(np.median(source_points[:, 1]))
    target_median_y = float(np.median(target_points[:, 1]))

    candidates: list[dict[str, Any]] = []
    candidate_id = 1
    h_values = np.linspace(args.topview_scale_min, args.topview_scale_max, args.topview_scale_steps)
    yaw_values = np.arange(-180.0, 180.0 + 0.5 * args.topview_yaw_step_deg, args.topview_yaw_step_deg)
    for yaw in yaw_values:
        rotation = _yaw_rotation(float(yaw))
        for horizontal_scale in h_values:
            started = time.perf_counter()
            scaled_rotated = (rotation @ np.diag([horizontal_scale, 1.0, horizontal_scale]) @ source_points.T).T
            source_hist = _topview_histogram(scaled_rotated, mins, shape, resolution)
            corr = np.fft.irfftn(target_fft * np.conj(np.fft.rfftn(source_hist)), s=shape, axes=(0, 1))
            peak_index = np.unravel_index(int(np.argmax(corr)), corr.shape)
            peak_value = float(corr[peak_index])
            shift = np.asarray(peak_index, dtype=float)
            shift = np.where(shift > np.asarray(shape, dtype=float) / 2.0, shift - np.asarray(shape, dtype=float), shift)
            translation_xz = shift * resolution
            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[0, 3] = float(translation_xz[0])
            matrix[1, 3] = target_median_y - source_median_y
            matrix[2, 3] = float(translation_xz[1])
            occupancy = float(np.count_nonzero(source_hist))
            peak_ratio = peak_value / max(occupancy, 1.0)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "status": "success",
                    "coarse_method": "topview_yaw_scale_correlation",
                    "topview_yaw_deg": float(yaw),
                    "topview_horizontal_scale": float(horizontal_scale),
                    "topview_peak": peak_value,
                    "topview_peak_ratio": peak_ratio,
                    "algorithm_fitness": peak_ratio,
                    "algorithm_inlier_rmse": 1.0 / max(peak_value, 1.0),
                    "algorithm_correspondence_set_size": int(peak_value),
                    "coarse_score": peak_ratio + 0.0008 * peak_value,
                    "runtime": time.perf_counter() - started,
                    "_matrix": matrix,
                }
            )
            candidate_id += 1
    return sorted(candidates, key=lambda r: float(r.get("coarse_score", float("-inf"))), reverse=True)[: args.topview_candidates]


def _topview_histogram(points: np.ndarray, mins: np.ndarray, shape: np.ndarray, resolution: float) -> np.ndarray:
    indices = np.floor((points[:, [0, 2]] - mins) / resolution).astype(int)
    mask = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < int(shape[0]))
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < int(shape[1]))
    )
    hist = np.zeros((int(shape[0]), int(shape[1])), dtype=float)
    if np.any(mask):
        valid = indices[mask]
        hist[valid[:, 0], valid[:, 1]] = 1.0
    return hist


def _yaw_rotation(deg: float) -> np.ndarray:
    angle = math.radians(float(deg))
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def _cluster_by_rotation(candidates: list[dict[str, Any]], threshold_deg: float) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda r: float(r.get("coarse_score", float("-inf"))), reverse=True):
        placed = False
        for cluster in clusters:
            if _rotation_diff_deg(item["_matrix"], cluster["rep"]["_matrix"]) <= threshold_deg:
                cluster["items"].append(item)
                placed = True
                break
        if not placed:
            clusters.append({"cluster_id": len(clusters) + 1, "rep": item, "items": [item]})
    return sorted(clusters, key=lambda c: len(c["items"]), reverse=True)


def _select_for_refine(clusters: list[dict[str, Any]], top_score_ids: set[int], top_clusters: int, max_candidates: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for cluster in clusters[:top_clusters]:
        top_items = sorted(cluster["items"], key=lambda r: float(r.get("coarse_score", float("-inf"))), reverse=True)[:1]
        for item in top_items:
            _append_unique(selected, seen, item)
            if len(selected) >= max_candidates:
                return selected
    for cluster in clusters:
        for item in cluster["items"]:
            if int(item["candidate_id"]) in top_score_ids:
                _append_unique(selected, seen, item)
            if len(selected) >= max_candidates:
                return selected
    return selected[:max_candidates]


def _append_unique(selected: list[dict[str, Any]], seen: set[int], item: dict[str, Any]) -> None:
    candidate_id = int(item["candidate_id"])
    if candidate_id not in seen:
        selected.append(item)
        seen.add(candidate_id)


def _scale_vote_refine(o3d: Any, source_down: Any, target_down: Any, candidate: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    coarse = np.asarray(candidate["_matrix"], dtype=float)
    scaled_matrix, scale_values, vote_info = _scale_translation_vote(source_down, target_down, coarse, args)
    gicp_started = time.perf_counter()
    final_matrix, gicp_status, gicp_reason, drift = _bounded_gicp(o3d, source_down, target_down, coarse, scaled_matrix, scale_values, args)
    gicp_time = time.perf_counter() - gicp_started
    eval_metrics = metrics.evaluate_registration(source_down, target_down, final_matrix, {"trimmed_ratio": 0.9, "overlap_threshold": args.eval_overlap_threshold})
    record = {
        "candidate_id": int(candidate["candidate_id"]),
        "cluster_id": int(candidate["cluster_id"]),
        "cluster_size": int(candidate["cluster_size"]),
        "cluster_top_score_count": int(candidate["cluster_top_score_count"]),
        "coarse_score": candidate.get("coarse_score"),
        "algorithm_fitness": candidate.get("algorithm_fitness"),
        "algorithm_inlier_rmse": candidate.get("algorithm_inlier_rmse"),
        "scale_values": scale_values,
        "vote_peak_count": vote_info["peak_count"],
        "vote_peak_ratio": vote_info["peak_ratio"],
        "vote_inlier_count": vote_info["inlier_count"],
        "scale_vote_error": vote_info["error"],
        "gicp_status": gicp_status,
        "gicp_reject_reason": gicp_reason,
        "gicp_rotation_drift_deg": drift["rotation_deg"],
        "gicp_translation_drift": drift["translation"],
        "gicp_time": gicp_time,
        "candidate_total_time": time.perf_counter() - started,
        "_coarse_matrix": coarse,
        "_scaled_matrix": scaled_matrix,
        "_final_matrix": final_matrix,
    }
    record.update(eval_metrics)
    record["refined_rank_score"] = _rank_score(record)
    return record


def _scale_translation_vote(source_down: Any, target_down: Any, coarse: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    source_points = np.asarray(source_down.points, dtype=float)
    if len(source_points) > args.max_vote_points:
        idx = np.linspace(0, len(source_points) - 1, args.max_vote_points).astype(int)
        source_points = source_points[idx]
    target_points = np.asarray(target_down.points, dtype=float)
    tree = _target_tree(target_down)
    rotation = coarse[:3, :3]
    h_values = np.linspace(args.horizontal_scale_min, args.horizontal_scale_max, args.horizontal_scale_steps)
    v_values = np.linspace(args.vertical_scale_min, args.vertical_scale_max, args.vertical_scale_steps)
    best = {
        "matrix": coarse.copy(),
        "scale": {"sx": 1.0, "sy": 1.0, "sz": 1.0},
        "error": float("inf"),
        "peak_count": 0,
        "peak_ratio": 0.0,
        "inlier_count": 0,
    }
    for h in h_values:
        for v in v_values:
            scaled_rotated = (rotation @ np.diag([h, v, h]) @ source_points.T).T
            votes = []
            distances = []
            for point in scaled_rotated:
                count, idx, squared = tree.search_knn_vector_3d(point + coarse[:3, 3], 1)
                if count:
                    dist = float(np.sqrt(squared[0]))
                    if dist <= args.vote_pair_distance:
                        votes.append(target_points[idx[0]] - point)
                        distances.append(dist)
            if len(votes) < args.min_vote_pairs:
                continue
            votes_array = np.asarray(votes, dtype=float)
            bins = np.floor(votes_array / args.translation_vote_voxel).astype(int)
            unique, counts = np.unique(bins, axis=0, return_counts=True)
            best_bin = unique[int(np.argmax(counts))]
            bin_mask = np.all(bins == best_bin, axis=1)
            if int(np.sum(bin_mask)) < args.min_vote_peak:
                continue
            local_votes = votes_array[bin_mask]
            translation = np.median(local_votes, axis=0)
            residual = np.linalg.norm(local_votes - translation, axis=1)
            keep = np.argsort(residual)[: max(4, int(len(residual) * args.vote_trimmed_ratio))]
            error = float(np.mean(residual[keep]))
            peak_count = int(np.sum(bin_mask))
            peak_ratio = float(peak_count / max(len(votes), 1))
            score_tuple = (peak_count, peak_ratio, -error)
            best_tuple = (best["peak_count"], best["peak_ratio"], -best["error"])
            if score_tuple > best_tuple:
                matrix = np.eye(4)
                matrix[:3, :3] = rotation @ np.diag([h, v, h])
                matrix[:3, 3] = translation
                best = {
                    "matrix": matrix,
                    "scale": {"sx": float(h), "sy": float(v), "sz": float(h)},
                    "error": error,
                    "peak_count": peak_count,
                    "peak_ratio": peak_ratio,
                    "inlier_count": int(len(votes)),
                }
    return best["matrix"], best["scale"], {k: best[k] for k in ("error", "peak_count", "peak_ratio", "inlier_count")}


def _target_tree(target_pcd: Any) -> Any:
    import open3d as o3d

    return o3d.geometry.KDTreeFlann(target_pcd)


def _bounded_gicp(o3d: Any, source_down: Any, target_down: Any, coarse: np.ndarray, scaled_matrix: np.ndarray, scale_values: Mapping[str, float], args: argparse.Namespace) -> tuple[np.ndarray, str, str | None, dict[str, float]]:
    if args.gicp_max_iterations <= 0:
        return scaled_matrix, "disabled", None, {"rotation_deg": 0.0, "translation": 0.0}
    scale = np.eye(4)
    scale[:3, :3] = np.diag([scale_values["sx"], scale_values["sy"], scale_values["sz"]])
    scaled_source = copy.deepcopy(source_down)
    scaled_source.transform(scale)
    init = np.eye(4)
    init[:3, :3] = coarse[:3, :3]
    init[:3, 3] = scaled_matrix[:3, 3]
    try:
        registration = o3d.pipelines.registration
        result = registration.registration_generalized_icp(
            scaled_source,
            target_down,
            args.gicp_max_correspondence_distance,
            init,
            registration.TransformationEstimationForGeneralizedICP(),
            registration.ICPConvergenceCriteria(max_iteration=args.gicp_max_iterations),
        )
        refined = np.asarray(result.transformation, dtype=float)
        rotation_drift = _rotation_angle_deg(refined[:3, :3] @ coarse[:3, :3].T)
        translation_drift = float(np.linalg.norm(refined[:3, 3] - init[:3, 3]))
        drift = {"rotation_deg": rotation_drift, "translation": translation_drift}
        if rotation_drift > args.bound_rotation_deg:
            return scaled_matrix, "rejected", "rotation_drift_exceeded", drift
        if translation_drift > args.bound_translation:
            return scaled_matrix, "rejected", "translation_drift_exceeded", drift
        return refined @ scale, "accepted", None, drift
    except Exception as exc:
        return scaled_matrix, "failed", str(exc), {"rotation_deg": 0.0, "translation": 0.0}


def _rank_score(record: Mapping[str, Any]) -> float:
    fitness = float(record.get("eval_fitness") or 0.0)
    trimmed = float(record.get("eval_trimmed_mean_nn_dist") or 10.0)
    rmse = float(record.get("eval_inlier_rmse") or 10.0)
    vote_ratio = float(record.get("vote_peak_ratio") or 0.0)
    return fitness - 0.3 * trimmed - 0.12 * rmse + 0.25 * vote_ratio + 0.06 * float(record.get("coarse_score") or 0.0)


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
    candidates: list[dict[str, Any]],
    records: list[dict[str, Any]],
    ranked_clusters: list[dict[str, Any]],
    source_raw: Any,
    target_raw: Any,
    args: argparse.Namespace,
    timings: Mapping[str, float],
) -> None:
    serializable_candidates = [{k: v for k, v in item.items() if not k.startswith("_")} for item in candidates]
    serializable_records = [{k: v for k, v in item.items() if not k.startswith("_")} for item in records]
    _write_json(run_dir / "candidate_metrics.json", serializable_candidates)
    _write_json(run_dir / "metrics.json", serializable_records)
    _write_json(run_dir / "cluster_summary.json", ranked_clusters)
    _write_csv(run_dir / "metrics.csv", serializable_records)
    record_by_id = {int(record["candidate_id"]): record for record in records}
    selected = []
    for cluster in ranked_clusters[: args.visualize_clusters]:
        candidate_id = cluster.get("best_candidate_id")
        if candidate_id is None:
            continue
        record = record_by_id[int(candidate_id)]
        matrix_path = run_dir / "matrix" / f"cluster_{int(cluster['cluster_id']):03d}__candidate_{int(candidate_id):04d}__fast.txt"
        io.write_matrix(matrix_path, record["_final_matrix"])
        record["matrix_path"] = str(matrix_path)
        stem = f"cluster_{int(cluster['cluster_id']):03d}__candidate_{int(candidate_id):04d}"
        cloud_path = run_dir / "cloud" / f"{stem}__registered.ply"
        overlay_path = run_dir / "cloud" / f"{stem}__overlay.ply"
        visualization.save_transformed_cloud(source_raw, record["_final_matrix"], cloud_path)
        visualization.save_overlay_cloud(source_raw, target_raw, record["_final_matrix"], overlay_path)
        record["cloud_path"] = str(cloud_path)
        record["overlay_path"] = str(overlay_path)
        selected.append(record)
    _write_json(run_dir / "selected_top_clusters.json", [{k: v for k, v in item.items() if not k.startswith("_")} for item in selected])
    _write_summary(run_dir, ranked_clusters, selected, args, timings)
    _write_visualization_commands(run_dir / "report" / "top_cluster_visualization_commands.md", selected)


def _write_summary(run_dir: Path, ranked_clusters: list[dict[str, Any]], selected: list[dict[str, Any]], args: argparse.Namespace, timings: Mapping[str, float]) -> None:
    lines = [
        "# Fast Rotation Cluster Scale Vote",
        "",
        f"- candidate_mode: `{args.candidate_mode}`",
        f"- trials: `{args.trials}`",
        f"- topview_candidates: `{args.topview_candidates}`",
        f"- top_clusters: `{args.top_clusters}`",
        f"- top_score_candidates: `{args.top_score_candidates}`",
        f"- max_refine_candidates: `{args.max_refine_candidates}`",
        f"- preprocess_time: `{timings['preprocess_time']:.6f}`",
        f"- coarse_time: `{timings['coarse_time']:.6f}`",
        f"- refine_time: `{timings['refine_time']:.6f}`",
        f"- total_time: `{timings['total_time']:.6f}`",
        "",
        "## Ranked Clusters",
        "",
        "| rank | cluster | size | top_count | best_candidate | best_score | fitness | trimmed | reference_family |",
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
                f"- matrix: `{record.get('matrix_path')}`",
                f"- scale_values: `{record.get('scale_values')}`",
                f"- vote_peak_ratio: `{_fmt(record.get('vote_peak_ratio'))}`",
                f"- gicp_status: `{record.get('gicp_status')}`",
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
    lines = ["# Fast Top Cluster Visualization Commands", ""]
    for record in records:
        lines.extend(
            [
                f"## cluster {record.get('cluster_id')} / candidate {record.get('candidate_id')}",
                "",
                f"- is_reference_family: `{record.get('is_reference_family')}`",
                f"- scale_values: `{record.get('scale_values')}`",
                "",
                "```bat",
                ".env\\python.exe scripts\\visualize_overlay.py --source \"{source}\" --target \"{target}\" --matrix \"{matrix}\"".format(
                    source=SOURCE["path"],
                    target=TARGET["path"],
                    matrix=record.get("matrix_path"),
                ),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


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
    parser = argparse.ArgumentParser(description="Fast RANSAC rotation-cluster scale-vote registration experiment.")
    parser.add_argument("--config", default="configs/ransac_all_no_refine.yaml")
    parser.add_argument("--candidate-mode", choices=["topview", "ransac"], default="topview")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--voxel-size", type=float, default=0.8)
    parser.add_argument("--distance-threshold-factor", type=float, default=2.25)
    parser.add_argument("--ransac-n", type=int, default=4)
    parser.add_argument("--max-iteration", type=int, default=60000)
    parser.add_argument("--confidence", type=float, default=0.995)
    parser.add_argument("--topview-grid-resolution", type=float, default=0.5)
    parser.add_argument("--topview-padding", type=float, default=8.0)
    parser.add_argument("--topview-yaw-step-deg", type=float, default=5.0)
    parser.add_argument("--topview-scale-min", type=float, default=0.65)
    parser.add_argument("--topview-scale-max", type=float, default=1.25)
    parser.add_argument("--topview-scale-steps", type=int, default=13)
    parser.add_argument("--topview-candidates", type=int, default=200)
    parser.add_argument("--cluster-rotation-deg", type=float, default=5.0)
    parser.add_argument("--top-clusters", type=int, default=27)
    parser.add_argument("--top-score-candidates", type=int, default=16)
    parser.add_argument("--max-refine-candidates", type=int, default=27)
    parser.add_argument("--visualize-clusters", type=int, default=5)
    parser.add_argument("--horizontal-scale-min", type=float, default=0.75)
    parser.add_argument("--horizontal-scale-max", type=float, default=1.2)
    parser.add_argument("--horizontal-scale-steps", type=int, default=5)
    parser.add_argument("--vertical-scale-min", type=float, default=0.7)
    parser.add_argument("--vertical-scale-max", type=float, default=1.15)
    parser.add_argument("--vertical-scale-steps", type=int, default=4)
    parser.add_argument("--max-vote-points", type=int, default=240)
    parser.add_argument("--vote-pair-distance", type=float, default=4.0)
    parser.add_argument("--translation-vote-voxel", type=float, default=0.35)
    parser.add_argument("--min-vote-pairs", type=int, default=10)
    parser.add_argument("--min-vote-peak", type=int, default=3)
    parser.add_argument("--vote-trimmed-ratio", type=float, default=0.7)
    parser.add_argument("--gicp-max-correspondence-distance", type=float, default=0.8)
    parser.add_argument("--gicp-max-iterations", type=int, default=0)
    parser.add_argument("--bound-rotation-deg", type=float, default=2.0)
    parser.add_argument("--bound-translation", type=float, default=0.35)
    parser.add_argument("--eval-overlap-threshold", type=float, default=1.0)
    parser.add_argument("--reference-matrix", default=REFERENCE_MATRIX)
    parser.add_argument("--correct-rotation-deg", type=float, default=5.0)
    args = parser.parse_args()
    run_dir, records, ranked_clusters = run_experiment(io.read_config(args.config), args)
    print(f"wrote {len(records)} refined candidates to {run_dir}")
    for idx, cluster in enumerate(ranked_clusters[:3], start=1):
        print(
            "rank={rank} cluster={cluster} size={size} top_count={top_count} best_candidate={candidate} reference_family={ref_family}".format(
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

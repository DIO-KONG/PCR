from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from utils.metrics import evaluate_registration, matrix_diagnostics, quality_status
from utils.pointcloud import clone_transform, read_point_cloud
from utils.preprocess import prepare_point_cloud
from utils.types import Candidate, RegistrationResult


def yaw_rotation(deg: float) -> np.ndarray:
    angle = np.deg2rad(deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def topview_hist(points: np.ndarray, mins: np.ndarray, shape: tuple[int, int], resolution: float) -> np.ndarray:
    ij = np.floor((points[:, [0, 2]] - mins) / resolution).astype(int)
    mask = (
        (ij[:, 0] >= 0)
        & (ij[:, 0] < shape[0])
        & (ij[:, 1] >= 0)
        & (ij[:, 1] < shape[1])
    )
    hist = np.zeros(shape, dtype=np.float32)
    valid = ij[mask]
    hist[valid[:, 0], valid[:, 1]] = 1.0
    return hist


def generate_topview_candidates(source_down, target_down, config: dict) -> list[Candidate]:
    source_points = np.asarray(source_down.points)
    target_points = np.asarray(target_down.points)
    resolution = float(config["topview_grid_resolution"])
    padding = float(config["topview_padding"])
    mins = np.minimum(source_points[:, [0, 2]].min(0), target_points[:, [0, 2]].min(0)) - padding
    maxs = np.maximum(source_points[:, [0, 2]].max(0), target_points[:, [0, 2]].max(0)) + padding
    shape_arr = np.ceil((maxs - mins) / resolution).astype(int) + 1
    shape = (int(shape_arr[0]), int(shape_arr[1]))

    target_hist = topview_hist(target_points, mins, shape, resolution)
    target_fft = np.fft.rfftn(target_hist)
    source_median_y = float(np.median(source_points[:, 1]))
    target_median_y = float(np.median(target_points[:, 1]))

    candidates: list[Candidate] = []
    cid = 1
    yaw_step = float(config["topview_yaw_step_deg"])
    yaws = np.arange(-180.0, 180.0 + yaw_step * 0.5, yaw_step)
    scales = np.linspace(
        float(config["topview_scale_min"]),
        float(config["topview_scale_max"]),
        int(config["topview_scale_steps"]),
    )

    for yaw in yaws:
        rotation = yaw_rotation(float(yaw))
        for horizontal_scale in scales:
            scaled_rotated = (rotation @ np.diag([horizontal_scale, 1.0, horizontal_scale]) @ source_points.T).T
            source_hist = topview_hist(scaled_rotated, mins, shape, resolution)
            corr = np.fft.irfftn(target_fft * np.conj(np.fft.rfftn(source_hist)), s=shape, axes=(0, 1))
            peak_idx = np.unravel_index(int(np.argmax(corr)), corr.shape)
            peak = float(corr[peak_idx])
            shift = np.asarray(peak_idx, dtype=float)
            shift = np.where(shift > np.asarray(shape) / 2.0, shift - np.asarray(shape), shift)
            t_xz = shift * resolution

            matrix = np.eye(4)
            matrix[:3, :3] = rotation
            matrix[0, 3] = t_xz[0]
            matrix[1, 3] = target_median_y - source_median_y
            matrix[2, 3] = t_xz[1]

            occupancy = max(float(np.count_nonzero(source_hist)), 1.0)
            peak_ratio = peak / occupancy
            score = peak_ratio + float(config.get("topview_peak_weight", 0.0008)) * peak
            candidates.append(
                Candidate(
                    candidate_id=cid,
                    matrix=matrix,
                    coarse_score=float(score),
                    metadata={
                        "topview_yaw_deg": float(yaw),
                        "topview_horizontal_scale": float(horizontal_scale),
                        "topview_peak": peak,
                        "topview_peak_ratio": peak_ratio,
                    },
                )
            )
            cid += 1

    limit = int(config.get("topview_candidates", 200))
    return sorted(candidates, key=lambda item: item.coarse_score, reverse=True)[:limit]


def rotation_angle_deg(rotation: np.ndarray) -> float:
    value = (np.trace(rotation) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(value, -1.0, 1.0))))


def cluster_by_rotation(candidates: list[Candidate], threshold_deg: float) -> list[dict]:
    clusters: list[dict] = []
    for item in sorted(candidates, key=lambda c: c.coarse_score, reverse=True):
        placed = False
        for cluster in clusters:
            diff = rotation_angle_deg(item.matrix[:3, :3] @ cluster["rep"].matrix[:3, :3].T)
            if diff <= threshold_deg:
                cluster["items"].append(item)
                placed = True
                break
        if not placed:
            clusters.append({"cluster_id": len(clusters) + 1, "rep": item, "items": [item]})
    return sorted(clusters, key=lambda c: len(c["items"]), reverse=True)


def select_candidates(candidates: list[Candidate], config: dict) -> list[Candidate]:
    clusters = cluster_by_rotation(candidates, float(config.get("cluster_rotation_deg", 5.0)))
    selected: list[Candidate] = []
    selected_ids: set[int] = set()
    top_clusters = int(config.get("top_clusters", 27))
    top_score_candidates = int(config.get("top_score_candidates", 16))
    max_refine_candidates = int(config.get("max_refine_candidates", 27))

    for cluster in clusters[:top_clusters]:
        best = max(cluster["items"], key=lambda item: item.coarse_score)
        best.cluster_id = int(cluster["cluster_id"])
        best.cluster_size = len(cluster["items"])
        best.cluster_top_score_count = 0
        selected.append(best)
        selected_ids.add(best.candidate_id)

    for item in sorted(candidates, key=lambda c: c.coarse_score, reverse=True)[:top_score_candidates]:
        if item.candidate_id not in selected_ids:
            selected.append(item)
            selected_ids.add(item.candidate_id)
        if item.cluster_top_score_count is None:
            item.cluster_top_score_count = 1

    return selected[:max_refine_candidates]


def scale_translation_vote(source_down, target_down, coarse: Candidate, config: dict) -> Candidate:
    source_points = np.asarray(source_down.points)
    if len(source_points) > int(config["max_vote_points"]):
        idx = np.linspace(0, len(source_points) - 1, int(config["max_vote_points"])).astype(int)
        source_points = source_points[idx]

    target_points = np.asarray(target_down.points)
    tree = o3d.geometry.KDTreeFlann(target_down)
    rotation = coarse.matrix[:3, :3]
    best: dict | None = None

    horizontal_scales = np.linspace(
        float(config["horizontal_scale_min"]),
        float(config["horizontal_scale_max"]),
        int(config["horizontal_scale_steps"]),
    )
    vertical_scales = np.linspace(
        float(config["vertical_scale_min"]),
        float(config["vertical_scale_max"]),
        int(config["vertical_scale_steps"]),
    )

    for horizontal_scale in horizontal_scales:
        for vertical_scale in vertical_scales:
            linear = rotation @ np.diag([horizontal_scale, vertical_scale, horizontal_scale])
            transformed_no_t = (linear @ source_points.T).T
            votes = []
            for point in transformed_no_t:
                count, indices, sq_distances = tree.search_knn_vector_3d(point + coarse.matrix[:3, 3], 1)
                if not count:
                    continue
                distance = float(np.sqrt(sq_distances[0]))
                if distance <= float(config["vote_pair_distance"]):
                    votes.append(target_points[indices[0]] - point)
            if len(votes) < int(config["min_vote_pairs"]):
                continue

            votes_arr = np.asarray(votes)
            bins = np.floor(votes_arr / float(config["translation_vote_voxel"])).astype(int)
            unique, counts = np.unique(bins, axis=0, return_counts=True)
            best_bin = unique[int(np.argmax(counts))]
            mask = np.all(bins == best_bin, axis=1)
            peak_count = int(mask.sum())
            if peak_count < int(config["min_vote_peak"]):
                continue

            local_votes = votes_arr[mask]
            translation = np.median(local_votes, axis=0)
            residual = np.linalg.norm(local_votes - translation, axis=1)
            keep_count = max(4, int(len(residual) * float(config["vote_trimmed_ratio"])))
            error = float(np.mean(np.sort(residual)[:keep_count]))
            peak_ratio = float(peak_count / max(len(votes_arr), 1))
            key = (peak_count, peak_ratio, -error)
            if best is None or key > best["key"]:
                matrix = np.eye(4)
                matrix[:3, :3] = linear
                matrix[:3, 3] = translation
                best = {
                    "key": key,
                    "matrix": matrix,
                    "scale_values": {
                        "sx": float(horizontal_scale),
                        "sy": float(vertical_scale),
                        "sz": float(horizontal_scale),
                    },
                    "vote_peak_ratio": peak_ratio,
                    "vote_peak_count": peak_count,
                    "vote_error": error,
                }

    if best is None:
        fallback = Candidate(
            candidate_id=coarse.candidate_id,
            matrix=coarse.matrix,
            coarse_score=coarse.coarse_score,
            cluster_id=coarse.cluster_id,
            cluster_size=coarse.cluster_size,
            cluster_top_score_count=coarse.cluster_top_score_count,
            scale_values={"sx": 1.0, "sy": 1.0, "sz": 1.0},
            vote_peak_ratio=0.0,
            metadata=dict(coarse.metadata),
        )
        fallback.metadata["vote_peak_count"] = 0
        return fallback

    refined = Candidate(
        candidate_id=coarse.candidate_id,
        matrix=best["matrix"],
        coarse_score=coarse.coarse_score,
        cluster_id=coarse.cluster_id,
        cluster_size=coarse.cluster_size,
        cluster_top_score_count=coarse.cluster_top_score_count,
        scale_values=best["scale_values"],
        vote_peak_ratio=best["vote_peak_ratio"],
        metadata=dict(coarse.metadata),
    )
    refined.metadata["vote_peak_count"] = best["vote_peak_count"]
    refined.metadata["vote_error"] = best["vote_error"]
    return refined


def rank_candidate(candidate: Candidate) -> float:
    metrics = candidate.metrics
    fitness = float(metrics.get("eval_fitness") or 0.0)
    trimmed = float(metrics.get("eval_trimmed_mean_nn_dist") or 999.0)
    rmse = float(metrics.get("eval_inlier_rmse") or 999.0)
    return (
        fitness
        - 0.3 * trimmed
        - 0.12 * rmse
        + 0.25 * candidate.vote_peak_ratio
        + 0.06 * candidate.coarse_score
    )


def register(source_path: str | Path, target_path: str | Path, config: dict) -> RegistrationResult:
    source_path = Path(source_path)
    target_path = Path(target_path)
    source_raw = read_point_cloud(source_path)
    target_raw = read_point_cloud(target_path)
    preprocess_config = dict(config.get("preprocess", {}))
    registration_config = dict(config.get("registration", {}))
    gate = dict(config.get("gate", {}))

    source_prepared = prepare_point_cloud(source_raw, preprocess_config, compute_fpfh=False)
    target_prepared = prepare_point_cloud(target_raw, preprocess_config, compute_fpfh=False)
    coarse_candidates = generate_topview_candidates(source_prepared.down, target_prepared.down, registration_config)
    selected = select_candidates(coarse_candidates, registration_config)

    refined: list[Candidate] = []
    for candidate in selected:
        item = scale_translation_vote(source_prepared.down, target_prepared.down, candidate, registration_config)
        transformed_down = clone_transform(source_prepared.down, item.matrix)
        item.metrics = {
            **evaluate_registration(
                transformed_down,
                target_prepared.down,
                overlap_threshold=float(registration_config["eval_overlap_threshold"]),
                trimmed_ratio=float(registration_config["eval_trimmed_ratio"]),
            ),
            **matrix_diagnostics(item.matrix),
        }
        item.rank_score = rank_candidate(item)
        item.metadata["status_hint"] = quality_status(item.metrics, gate)
        refined.append(item)

    refined = sorted(refined, key=lambda item: item.rank_score, reverse=True)
    if not refined:
        raise RuntimeError("No registration candidates were produced.")

    best = refined[0]
    metrics = dict(best.metrics)
    metrics["rank_score"] = best.rank_score
    metrics["coarse_score"] = best.coarse_score
    metrics["vote_peak_ratio"] = best.vote_peak_ratio
    metrics["status"] = quality_status(metrics, gate)
    return RegistrationResult(
        algorithm="topview_vote",
        source_path=source_path,
        target_path=target_path,
        matrix=best.matrix,
        metrics=metrics,
        status=str(metrics["status"]),
        top_candidates=refined[: int(config.get("output", {}).get("top_candidates", 5))],
    )

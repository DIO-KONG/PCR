from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from algorithm.base import RegistrationCandidate, RegistrationResult
from utils.pointcloud import load_cloud


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """读取 DA3 batch npz，并转成普通 dict，避免文件句柄生命周期问题。"""

    data = np.load(Path(path), allow_pickle=False)
    return {key: np.asarray(data[key]) for key in data.files}


def image_name_list(batch: dict[str, np.ndarray]) -> list[str]:
    """把 npz 中的 image_names 统一转成 Python 字符串列表。"""

    return [str(item) for item in batch["image_names"]]


def valid_conf_mask(conf: np.ndarray | None, percentile: float) -> np.ndarray | None:
    """根据 confidence percentile 生成高置信 mask。

    `percentile=20` 表示丢弃最低 20% confidence，保留较高的 80%。
    """

    if conf is None:
        return None
    finite = conf[np.isfinite(conf)]
    if finite.size == 0:
        return np.zeros_like(conf, dtype=bool)
    threshold = np.percentile(finite, float(percentile))
    return np.isfinite(conf) & (conf >= threshold)


def pixels_to_world(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """把像素深度反投影到 DA3 batch 局部 world 坐标。

    DA3 的 extrinsics 是 world-to-camera，因此这里取逆得到 camera-to-world。
    """

    z = depth[rows, cols].astype(np.float64)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise ValueError(f"Invalid intrinsic with zero focal length: {intrinsic}")

    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    camera_points = np.column_stack([x, y, z, np.ones_like(z)])
    camera_to_world = np.linalg.inv(extrinsic_w2c)
    return (camera_to_world @ camera_points.T).T[:, :3]


def sample_shared_frame_points(
    source_batch: dict[str, np.ndarray],
    target_batch: dict[str, np.ndarray],
    source_index: int,
    target_index: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """从一张共享帧中提取 source->target 的 3D 对应点。"""

    depth_s = source_batch["depth"][source_index]
    depth_t = target_batch["depth"][target_index]
    if depth_s.shape != depth_t.shape:
        raise ValueError(f"Shared frame depth shapes differ: source={depth_s.shape}, target={depth_t.shape}")

    sampling = config.get("sampling", {})
    stride = int(sampling.get("stride", 6))
    conf_percentile = float(sampling.get("conf_percentile", 20.0))
    max_points = int(sampling.get("max_points_per_frame", 8000))
    random_seed = int(sampling.get("random_seed", 7))

    valid = np.isfinite(depth_s) & np.isfinite(depth_t) & (depth_s > 0.0) & (depth_t > 0.0)
    conf_s = source_batch.get("conf")
    conf_t = target_batch.get("conf")
    mask_s = None if conf_s is None else valid_conf_mask(conf_s[source_index], conf_percentile)
    mask_t = None if conf_t is None else valid_conf_mask(conf_t[target_index], conf_percentile)
    if mask_s is not None:
        valid &= mask_s
    if mask_t is not None:
        valid &= mask_t

    if stride > 1:
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[::stride, ::stride] = True
        valid &= stride_mask

    rows, cols = np.nonzero(valid)
    if rows.size > max_points:
        # DA3 深度点在图像网格上有明显扫描线结构。这里使用固定随机种子采样，
        # 避免 linspace 采样长期偏向某些图像区域，同时保证实验可复现。
        rng = np.random.default_rng(random_seed + int(source_index) * 1009 + int(target_index))
        indices = rng.choice(rows.size, size=max_points, replace=False)
        rows = rows[indices]
        cols = cols[indices]

    source_points = pixels_to_world(
        depth_s,
        source_batch["intrinsics"][source_index],
        source_batch["extrinsics"][source_index],
        rows,
        cols,
    )
    target_points = pixels_to_world(
        depth_t,
        target_batch["intrinsics"][target_index],
        target_batch["extrinsics"][target_index],
        rows,
        cols,
    )
    return source_points, target_points


def collect_correspondences(
    source_batch: dict[str, np.ndarray],
    target_batch: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """收集所有共享帧的像素级 3D 对应关系。"""

    source_names = image_name_list(source_batch)
    target_names = image_name_list(target_batch)
    configured = config.get("shared_frames")
    if configured:
        shared_frames = [str(item) for item in configured]
    else:
        shared_frames = sorted(set(source_names) & set(target_names))

    source_all: list[np.ndarray] = []
    target_all: list[np.ndarray] = []
    frame_all: list[np.ndarray] = []
    frame_stats: list[dict[str, Any]] = []
    for name in shared_frames:
        if name not in source_names or name not in target_names:
            raise ValueError(f"Configured shared frame is not in both batches: {name}")
        source_index = source_names.index(name)
        target_index = target_names.index(name)
        source_points, target_points = sample_shared_frame_points(
            source_batch,
            target_batch,
            source_index,
            target_index,
            config,
        )
        frame_stats.append(
            {
                "frame": name,
                "source_index": source_index,
                "target_index": target_index,
                "correspondence_count": int(len(source_points)),
            }
        )
        if len(source_points):
            source_all.append(source_points)
            target_all.append(target_points)
            frame_all.append(np.full(len(source_points), name, dtype=object))

    if not source_all:
        raise RuntimeError("No valid shared-frame correspondences were collected.")
    return np.vstack(source_all), np.vstack(target_all), np.concatenate(frame_all), frame_stats


def estimate_rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """用 Kabsch/SVD 估计无尺度刚体变换 target ~= T @ source。"""

    if len(source) < 3:
        raise ValueError("At least 3 correspondences are required for rigid transform.")
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """应用 4x4 变换到 Nx3 点。"""

    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (matrix @ homogeneous.T).T[:, :3]


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """计算 3x3 旋转矩阵对应的旋转角，单位为度。"""

    trace = float(np.trace(rotation))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def error_metrics(errors: np.ndarray, threshold: float) -> dict[str, float]:
    """统计对应点误差。"""

    return {
        "inlier_threshold": float(threshold),
        "inlier_count": int(np.count_nonzero(errors <= threshold)),
        "inlier_ratio": float(np.mean(errors <= threshold)) if len(errors) else 0.0,
        "rmse": float(np.sqrt(np.mean(errors * errors))) if len(errors) else float("inf"),
        "median_error": float(np.median(errors)) if len(errors) else float("inf"),
        "p90_error": float(np.percentile(errors, 90)) if len(errors) else float("inf"),
        "mean_error": float(np.mean(errors)) if len(errors) else float("inf"),
    }


def score_metrics(metrics: dict[str, Any]) -> float:
    """把严格评估指标压成单个候选分数。

    分数只用于候选排序，不代表几何真值。这里故意惩罚 median/p90 误差，
    避免仅靠较宽松 inlier 数量把局部错误解排到最前。
    """

    return float(metrics["inlier_ratio"] - metrics["median_error"] - 0.25 * metrics["p90_error"])


def frame_error_metrics(
    source: np.ndarray,
    target: np.ndarray,
    frame_names: np.ndarray,
    matrix: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    """按共享帧拆分误差，便于定位某一帧外参/深度是否拖累整体估计。"""

    transformed = transform_points(source, matrix)
    errors = np.linalg.norm(transformed - target, axis=1)
    rows: list[dict[str, Any]] = []
    for frame in sorted({str(item) for item in frame_names}):
        mask = frame_names == frame
        frame_errors = errors[mask]
        metrics = error_metrics(frame_errors, threshold)
        rows.append(
            {
                "frame": frame,
                "correspondence_count": int(np.count_nonzero(mask)),
                **metrics,
            }
        )
    return rows


def evaluate_shared_correspondences(
    source: np.ndarray,
    target: np.ndarray,
    frame_names: np.ndarray,
    matrix: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """用统一阈值评估某个矩阵在共享帧对应点上的表现。"""

    errors = np.linalg.norm(transform_points(source, matrix) - target, axis=1)
    metrics: dict[str, Any] = error_metrics(errors, threshold)
    metrics["per_frame_metrics"] = frame_error_metrics(source, target, frame_names, matrix, threshold)
    return metrics


def bounded_icp_refinement(matrix: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """在共享帧初值附近做有边界的 point-to-plane ICP。

    这是一个局部刚体精修步骤：Open3D ICP 只估计 SE(3) 变换，不引入尺度。
    为了避免相似隔板把结果拉走，ICP 后会检查相对初值的位移和旋转幅度。
    """

    icp_config = config.get("icp_refinement", {})
    if not bool(icp_config.get("enabled", False)):
        return matrix, {"enabled": False, "status": "disabled"}

    source_cloud_path = Path(config.get("source_cloud", ""))
    target_cloud_path = Path(config.get("target_cloud", ""))
    if not source_cloud_path or not target_cloud_path:
        return matrix, {"enabled": True, "status": "skipped_missing_cloud_paths"}

    voxel_size = float(icp_config.get("voxel_size", 0.06))
    max_distance = float(icp_config.get("max_correspondence_distance", 0.08))
    normal_radius = float(icp_config.get("normal_radius", voxel_size * 3.0))
    normal_max_nn = int(icp_config.get("normal_max_nn", 30))
    max_iteration = int(icp_config.get("max_iteration", 40))
    max_translation_delta = float(icp_config.get("max_translation_delta", 0.15))
    max_rotation_delta_deg = float(icp_config.get("max_rotation_delta_deg", 3.0))

    source_cloud = load_cloud(source_cloud_path)
    target_cloud = load_cloud(target_cloud_path)
    source_down = source_cloud.voxel_down_sample(voxel_size)
    target_down = target_cloud.voxel_down_sample(voxel_size)
    if source_down.is_empty() or target_down.is_empty():
        return matrix, {"enabled": True, "status": "skipped_empty_downsample"}

    # point-to-plane ICP 要求 target 有法线；source 法线也估计，便于后续切换到其他估计器。
    search = o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    source_down.estimate_normals(search)
    target_down.estimate_normals(search)

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        max_distance,
        matrix,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration),
    )

    refined = np.asarray(result.transformation, dtype=np.float64)
    delta = refined @ np.linalg.inv(matrix)
    translation_delta = float(np.linalg.norm(delta[:3, 3]))
    rotation_delta = rotation_angle_deg(delta[:3, :3])
    accepted = translation_delta <= max_translation_delta and rotation_delta <= max_rotation_delta_deg

    metrics = {
        "enabled": True,
        "status": "accepted" if accepted else "rejected_by_boundary",
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "max_correspondence_distance": max_distance,
        "voxel_size": voxel_size,
        "translation_delta": translation_delta,
        "rotation_delta_deg": rotation_delta,
        "max_translation_delta": max_translation_delta,
        "max_rotation_delta_deg": max_rotation_delta_deg,
    }
    if not accepted:
        return matrix, metrics
    return refined, metrics


def refine_rigid_by_thresholds(
    source: np.ndarray,
    target: np.ndarray,
    matrix: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """在候选矩阵附近做逐阈值刚体精修。

    注意：每次更新仍然调用 Kabsch/SVD，只估计旋转和平移，不估计尺度。
    阈值从宽到严逐步收紧，相当于反复丢弃明显不一致的共享像素点。
    """

    refinement = config.get("refinement", {})
    if not bool(refinement.get("enabled", True)):
        return matrix, []

    thresholds = [float(item) for item in refinement.get("thresholds", [0.12, 0.10, 0.08, 0.06])]
    iterations_per_threshold = int(refinement.get("iterations_per_threshold", 2))
    min_points = int(refinement.get("min_points", 200))
    current = matrix
    history: list[dict[str, Any]] = []

    for threshold in thresholds:
        for iteration in range(iterations_per_threshold):
            errors = np.linalg.norm(transform_points(source, current) - target, axis=1)
            inliers = errors <= threshold
            inlier_count = int(np.count_nonzero(inliers))
            if inlier_count < min_points:
                history.append(
                    {
                        "threshold": float(threshold),
                        "iteration": int(iteration),
                        "status": "skipped_not_enough_inliers",
                        "inlier_count": inlier_count,
                    }
                )
                break

            current = estimate_rigid_transform(source[inliers], target[inliers])
            refined_errors = np.linalg.norm(transform_points(source, current) - target, axis=1)
            step_metrics = error_metrics(refined_errors, threshold)
            history.append(
                {
                    "threshold": float(threshold),
                    "iteration": int(iteration),
                    "status": "refit",
                    **step_metrics,
                }
            )

    return current, history


def ransac_rigid_candidates(
    source: np.ndarray,
    target: np.ndarray,
    frame_names: np.ndarray,
    config: dict[str, Any],
) -> list[RegistrationCandidate]:
    """多阈值 RANSAC + 逐步收紧 refit，生成一组刚体候选。"""

    ransac = config.get("ransac", {})
    iterations = int(ransac.get("iterations", 2000))
    sample_size = int(ransac.get("sample_size", 6))
    thresholds = [float(item) for item in ransac.get("thresholds", [ransac.get("inlier_threshold", 0.06)])]
    evaluation_threshold = float(ransac.get("evaluation_threshold", ransac.get("inlier_threshold", 0.06)))
    seed = int(ransac.get("seed", 7))
    min_inliers = max(sample_size, int(ransac.get("min_inliers", 50)))
    if len(source) < sample_size:
        raise RuntimeError(f"Not enough correspondences for RANSAC: {len(source)} < {sample_size}")

    rng = np.random.default_rng(seed)
    candidates: list[RegistrationCandidate] = []

    for candidate_id, threshold in enumerate(thresholds, 1):
        best_matrix: np.ndarray | None = None
        best_inliers: np.ndarray | None = None
        best_tuple = (-1, float("inf"), float("inf"))

        for _ in range(iterations):
            sample = rng.choice(len(source), size=sample_size, replace=False)
            matrix = estimate_rigid_transform(source[sample], target[sample])
            errors = np.linalg.norm(transform_points(source, matrix) - target, axis=1)
            inliers = errors <= threshold
            count = int(np.count_nonzero(inliers))
            median = float(np.median(errors[inliers])) if count else float("inf")
            rmse = float(np.sqrt(np.mean(errors[inliers] * errors[inliers]))) if count else float("inf")
            key = (count, -median, -rmse)
            if key > best_tuple:
                best_tuple = key
                best_matrix = matrix
                best_inliers = inliers

        if best_matrix is None or best_inliers is None or np.count_nonzero(best_inliers) < min_inliers:
            continue

        # 先用当前 RANSAC 阈值内点做一次刚体 refit，再进入更严格的阈值收紧。
        refit = estimate_rigid_transform(source[best_inliers], target[best_inliers])
        refined, refinement_history = refine_rigid_by_thresholds(source, target, refit, config)
        metrics = evaluate_shared_correspondences(source, target, frame_names, refined, evaluation_threshold)
        metrics.update(
            {
                "coarse_threshold": float(threshold),
                "evaluation_threshold": float(evaluation_threshold),
                "ransac_iterations": iterations,
                "ransac_sample_size": sample_size,
                "refit_inlier_count": int(np.count_nonzero(best_inliers)),
                "refinement_history": refinement_history,
            }
        )
        candidates.append(
            RegistrationCandidate(
                candidate_id=candidate_id,
                matrix=refined,
                score=score_metrics(metrics),
                metrics=metrics,
                metadata={"method": "shared_frame_rigid_threshold_refine"},
            )
        )

    if not candidates:
        raise RuntimeError("RANSAC failed to find a valid shared-frame transform.")

    return sorted(candidates, key=lambda item: item.score, reverse=True)


def register(source_path: str | Path, target_path: str | Path, config: dict[str, Any]) -> RegistrationResult:
    """基于共享 RGB-D 帧的 DA3 batch 坐标系对齐。

    `source_path` / `target_path` 是框架传入的点云路径；本算法真正估计矩阵时读取
    `source_npz` / `target_npz`，然后返回可作用于 source cloud 的刚体矩阵。
    """

    source_npz = Path(config["source_npz"])
    target_npz = Path(config["target_npz"])
    source_batch = load_npz(source_npz)
    target_batch = load_npz(target_npz)
    source_points, target_points, frame_names, frame_stats = collect_correspondences(source_batch, target_batch, config)

    candidates = ransac_rigid_candidates(source_points, target_points, frame_names, config)
    top_k = int(config.get("output", {}).get("top_k", len(candidates)))
    candidates = candidates[:top_k]
    best = candidates[0]
    evaluation_threshold = float(config.get("ransac", {}).get("evaluation_threshold", config.get("ransac", {}).get("inlier_threshold", 0.06)))
    pre_icp_metrics = dict(best.metrics)
    matrix, icp_metrics = bounded_icp_refinement(best.matrix, config)
    metrics = evaluate_shared_correspondences(source_points, target_points, frame_names, matrix, evaluation_threshold)

    # ICP 在重复隔板/工位结构中可能取得较高点云 fitness，但破坏共享帧对应关系。
    # 因此除了运动边界，还要用共享帧几何一致性兜底；失败则回退到 ICP 前矩阵。
    icp_config = config.get("icp_refinement", {})
    if icp_metrics.get("status") == "accepted":
        max_median_ratio = float(icp_config.get("max_shared_median_error_ratio", 1.10))
        min_inlier_ratio_ratio = float(icp_config.get("min_shared_inlier_ratio_ratio", 0.85))
        pre_median = float(pre_icp_metrics.get("median_error", float("inf")))
        pre_inlier_ratio = float(pre_icp_metrics.get("inlier_ratio", 0.0))
        median_ok = metrics["median_error"] <= pre_median * max_median_ratio
        inlier_ok = metrics["inlier_ratio"] >= pre_inlier_ratio * min_inlier_ratio_ratio
        if not (median_ok and inlier_ok):
            icp_metrics = {
                **icp_metrics,
                "status": "rejected_by_shared_consistency",
                "shared_median_error_after_icp": metrics["median_error"],
                "shared_median_error_before_icp": pre_median,
                "shared_inlier_ratio_after_icp": metrics["inlier_ratio"],
                "shared_inlier_ratio_before_icp": pre_inlier_ratio,
                "max_shared_median_error_ratio": max_median_ratio,
                "min_shared_inlier_ratio_ratio": min_inlier_ratio_ratio,
            }
            matrix = best.matrix
            metrics = evaluate_shared_correspondences(source_points, target_points, frame_names, matrix, evaluation_threshold)

    metrics.update(
        {
            "coarse_threshold": best.metrics.get("coarse_threshold"),
            "evaluation_threshold": evaluation_threshold,
            "ransac_iterations": best.metrics.get("ransac_iterations"),
            "ransac_sample_size": best.metrics.get("ransac_sample_size"),
            "refit_inlier_count": best.metrics.get("refit_inlier_count"),
            "refinement_history": best.metrics.get("refinement_history", []),
            "pre_icp_shared_metrics": pre_icp_metrics,
            "icp_refinement": icp_metrics,
        }
    )
    metrics.update(
        {
            "correspondence_count": int(len(source_points)),
            "shared_frames": frame_stats,
            "allow_scale": False,
            "source_npz": str(source_npz),
            "target_npz": str(target_npz),
            "status": "accepted" if metrics["inlier_ratio"] >= float(config.get("ransac", {}).get("min_inlier_ratio", 0.35)) else "needs_review_low_inlier_ratio",
        }
    )
    best.matrix = matrix
    best.metrics = metrics
    best.score = score_metrics(metrics)
    return RegistrationResult(
        algorithm="shared_frame_alignment",
        source_path=Path(source_path),
        target_path=Path(target_path),
        status=str(metrics["status"]),
        matrix=matrix,
        metrics=metrics,
        candidates=candidates,
    )

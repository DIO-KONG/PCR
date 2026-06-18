from __future__ import annotations

from typing import Any

import numpy as np
import open3d as o3d

from pcr.algorithms.shared_frame.geometry import rotation_angle_deg, transform_points
from pcr.domain import IcpMetrics, RefinementResult, Transform


class BoundedPointToPlaneIcp:
    """有边界 point-to-plane ICP。

    ICP 只允许在 coarse 初值附近做小幅 SE(3) 调整。若相对初值的平移或旋转
    超过配置边界，`accepted_transform` 会自动回退到初值。
    """

    def refine(
        self,
        *,
        source_cloud: o3d.geometry.PointCloud,
        target_world: o3d.geometry.PointCloud,
        initial_transform: Transform,
        icp_config: dict[str, Any],
    ) -> RefinementResult:
        voxel_size = float(icp_config["voxel_size"])
        source_down = source_cloud.voxel_down_sample(voxel_size)
        target_down = target_world.voxel_down_sample(voxel_size)
        if source_down.is_empty() or target_down.is_empty():
            raise RuntimeError("ICP downsampled source or target world is empty.")

        search = o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(icp_config["normal_radius"]),
            max_nn=int(icp_config["normal_max_nn"]),
        )
        source_down.estimate_normals(search)
        target_down.estimate_normals(search)

        result = o3d.pipelines.registration.registration_icp(
            source_down,
            target_down,
            float(icp_config["max_correspondence_distance"]),
            initial_transform.matrix,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(icp_config["max_iteration"])),
        )
        refined_matrix = np.asarray(result.transformation, dtype=float)
        delta_matrix = refined_matrix @ np.linalg.inv(initial_transform.matrix)
        translation_delta = float(np.linalg.norm(delta_matrix[:3, 3]))
        rotation_delta = rotation_angle_deg(delta_matrix[:3, :3])
        accepted = (
            translation_delta <= float(icp_config["max_translation_delta"])
            and rotation_delta <= float(icp_config["max_rotation_delta_deg"])
        )

        refined = Transform(
            source=initial_transform.source,
            target=initial_transform.target,
            matrix=refined_matrix,
        )
        delta = Transform(
            source=initial_transform.target,
            target=initial_transform.target,
            matrix=delta_matrix,
        )
        metrics = IcpMetrics(
            status="accepted" if accepted else "rejected_by_boundary",
            fitness=float(result.fitness),
            inlier_rmse=float(result.inlier_rmse),
            translation_delta=translation_delta,
            rotation_delta_deg=rotation_delta,
            extra={
                "max_translation_delta": float(icp_config["max_translation_delta"]),
                "max_rotation_delta_deg": float(icp_config["max_rotation_delta_deg"]),
                "max_correspondence_distance": float(icp_config["max_correspondence_distance"]),
                "voxel_size": voxel_size,
            },
        )
        return RefinementResult(
            initial_transform=initial_transform,
            refined_transform=refined,
            delta_from_initial=delta,
            accepted_transform=refined if accepted else initial_transform,
            accepted=accepted,
            metrics=metrics,
        )


def _estimate_sim3_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """估计 `target ~= scale * R @ source + t` 的 Sim(3) 变换。

    粗配准仍然只估计 SE(3)。这个函数只用于 refine 阶段吸收 DA3 不同 batch
    之间的小尺度差异，避免把尺度自由度扩散到 shared-frame coarse。
    """

    if len(source) < 3:
        raise ValueError("At least 3 correspondences are required for Sim(3) estimation.")
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

    rotated_source = (rotation @ source_centered.T).T
    denominator = float(np.sum(source_centered * source_centered))
    if denominator <= 0:
        raise ValueError("Degenerate source correspondences for Sim(3) estimation.")
    scale = float(np.sum(target_centered * rotated_source) / denominator)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid Sim(3) scale estimated: {scale}")
    translation = target_centroid - scale * rotation @ source_centroid

    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    return matrix


def _sim3_scale(matrix: np.ndarray) -> float:
    """从 4x4 Sim(3) 矩阵中提取均匀尺度。"""

    linear = np.asarray(matrix, dtype=float)[:3, :3]
    det = float(np.linalg.det(linear))
    return float(np.cbrt(abs(det)))


def _sim3_rotation(matrix: np.ndarray) -> np.ndarray:
    """从 Sim(3) 矩阵中提取去尺度旋转部分。"""

    scale = _sim3_scale(matrix)
    if scale <= 0:
        return np.eye(3, dtype=float)
    return np.asarray(matrix, dtype=float)[:3, :3] / scale


class BoundedSim3PointToPointIcp:
    """有边界 Sim(3) point-to-point refine。

    它用 coarse 的 source->global 作为初值，只允许在附近估计一个小尺度修正。
    若 refine 相对初值的平移、旋转或尺度变化超出边界，则回退到 coarse。
    """

    def refine(
        self,
        *,
        source_cloud: o3d.geometry.PointCloud,
        target_world: o3d.geometry.PointCloud,
        initial_transform: Transform,
        icp_config: dict[str, Any],
    ) -> RefinementResult:
        voxel_size = float(icp_config["voxel_size"])
        source_down = source_cloud.voxel_down_sample(voxel_size)
        target_down = target_world.voxel_down_sample(voxel_size)
        if source_down.is_empty() or target_down.is_empty():
            raise RuntimeError("Sim3 ICP downsampled source or target world is empty.")

        source_points = np.asarray(source_down.points, dtype=float)
        target_points = np.asarray(target_down.points, dtype=float)
        tree = o3d.geometry.KDTreeFlann(target_down)
        max_distance = float(icp_config["max_correspondence_distance"])
        max_distance2 = max_distance * max_distance
        min_correspondences = int(icp_config.get("min_correspondences", 50))
        current = np.asarray(initial_transform.matrix, dtype=float).copy()
        best_rmse = float("inf")
        best_count = 0

        for _ in range(int(icp_config["max_iteration"])):
            transformed = transform_points(source_points, current)
            source_inliers: list[int] = []
            matched_targets: list[np.ndarray] = []
            squared_errors: list[float] = []
            for index, point in enumerate(transformed):
                count, indices, distances = tree.search_knn_vector_3d(point, 1)
                if count and distances[0] <= max_distance2:
                    source_inliers.append(index)
                    matched_targets.append(target_points[indices[0]])
                    squared_errors.append(float(distances[0]))

            if len(source_inliers) < min_correspondences:
                break

            rmse = float(np.sqrt(np.mean(squared_errors)))
            if abs(best_rmse - rmse) < float(icp_config.get("sim3_convergence_rmse_delta", 1e-5)):
                best_rmse = rmse
                best_count = len(source_inliers)
                break
            best_rmse = rmse
            best_count = len(source_inliers)
            current = _estimate_sim3_transform(
                source_points[np.asarray(source_inliers, dtype=int)],
                np.asarray(matched_targets, dtype=float),
            )

        refined_matrix = current
        delta_matrix = refined_matrix @ np.linalg.inv(initial_transform.matrix)
        delta_scale = _sim3_scale(delta_matrix)
        translation_delta = float(np.linalg.norm(delta_matrix[:3, 3]))
        rotation_delta = rotation_angle_deg(_sim3_rotation(delta_matrix))
        scale_delta = abs(delta_scale - 1.0)
        max_scale_delta = float(icp_config.get("max_scale_delta", 0.08))
        accepted = (
            best_count >= min_correspondences
            and translation_delta <= float(icp_config["max_translation_delta"])
            and rotation_delta <= float(icp_config["max_rotation_delta_deg"])
            and scale_delta <= max_scale_delta
        )

        refined = Transform(
            source=initial_transform.source,
            target=initial_transform.target,
            matrix=refined_matrix,
        )
        delta = Transform(
            source=initial_transform.target,
            target=initial_transform.target,
            matrix=delta_matrix,
        )
        metrics = IcpMetrics(
            status="accepted" if accepted else "rejected_by_boundary",
            fitness=float(best_count / len(source_points)) if len(source_points) else 0.0,
            inlier_rmse=best_rmse if np.isfinite(best_rmse) else float("inf"),
            translation_delta=translation_delta,
            rotation_delta_deg=rotation_delta,
            extra={
                "method": "sim3_point_to_point",
                "scale_delta": scale_delta,
                "delta_scale": delta_scale,
                "max_scale_delta": max_scale_delta,
                "max_translation_delta": float(icp_config["max_translation_delta"]),
                "max_rotation_delta_deg": float(icp_config["max_rotation_delta_deg"]),
                "max_correspondence_distance": max_distance,
                "voxel_size": voxel_size,
                "correspondence_count": best_count,
                "min_correspondences": min_correspondences,
            },
        )
        return RefinementResult(
            initial_transform=initial_transform,
            refined_transform=refined,
            delta_from_initial=delta,
            accepted_transform=refined if accepted else initial_transform,
            accepted=accepted,
            metrics=metrics,
        )


def make_refiner(icp_config: dict[str, Any]):
    """按配置选择 refine 算法，默认保持原 point-to-plane ICP。"""

    method = str(icp_config.get("method", "point_to_plane_icp"))
    if method == "sim3_point_to_point":
        return BoundedSim3PointToPointIcp()
    if method == "point_to_plane_icp":
        return BoundedPointToPlaneIcp()
    raise ValueError(f"Unknown ICP/refinement method: {method}")

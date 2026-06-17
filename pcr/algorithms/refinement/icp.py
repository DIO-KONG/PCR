from __future__ import annotations

from typing import Any

import numpy as np
import open3d as o3d

from pcr.algorithms.shared_frame.geometry import rotation_angle_deg
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

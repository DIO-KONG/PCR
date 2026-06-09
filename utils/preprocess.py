from __future__ import annotations

from dataclasses import dataclass

import open3d as o3d


@dataclass
class PreparedCloud:
    """预处理后的点云集合。

    `raw` 保留原始点云用于最终 transform/overlay；
    `down` 用于注册和指标评估，避免在全量点上做昂贵搜索。
    """

    raw: o3d.geometry.PointCloud
    working: o3d.geometry.PointCloud
    down: o3d.geometry.PointCloud
    fpfh: o3d.pipelines.registration.Feature | None = None


def prepare_point_cloud(
    cloud: o3d.geometry.PointCloud,
    config: dict,
    *,
    compute_fpfh: bool = False,
) -> PreparedCloud:
    """点云预处理。

    当前流程：可选 SOR 去离群点 -> 体素降采样 -> 法线估计 -> 可选 FPFH。
    topview_vote 不需要 FPFH，RANSAC 对照算法需要。
    """

    working = cloud
    if config.get("sor_enabled", True):
        working, _ = working.remove_statistical_outlier(
            nb_neighbors=int(config.get("sor_nb_neighbors", 20)),
            std_ratio=float(config.get("sor_std_ratio", 2.0)),
        )

    voxel_size = float(config.get("voxel_size", config.get("registration_voxel_size", 0.1)))
    down = working.voxel_down_sample(voxel_size)

    normal_radius = voxel_size * float(config.get("normal_radius_factor", 2.0))
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=int(config.get("normal_max_nn", 30)),
        )
    )

    fpfh = None
    if compute_fpfh:
        fpfh_radius = voxel_size * float(config.get("fpfh_radius_factor", 5.0))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            down,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=fpfh_radius,
                max_nn=int(config.get("fpfh_max_nn", 100)),
            ),
        )

    return PreparedCloud(raw=cloud, working=working, down=down, fpfh=fpfh)

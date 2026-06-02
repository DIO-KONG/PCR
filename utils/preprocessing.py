from __future__ import annotations

from typing import Any, Mapping


def prepare_point_cloud(pcd: Any, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for preprocessing.") from exc

    config = dict(config or {})
    voxel_size = float(config.get("voxel_size", 0.05))
    working = pcd

    sor = dict(config.get("remove_statistical_outlier", {}))
    if sor.get("enabled", False):
        working, _ = working.remove_statistical_outlier(
            nb_neighbors=int(sor.get("nb_neighbors", 20)),
            std_ratio=float(sor.get("std_ratio", 2.0)),
        )

    roi = dict(config.get("roi", {}))
    if roi.get("enabled", False):
        min_bound = roi.get("min_bound")
        max_bound = roi.get("max_bound")
        if min_bound is not None and max_bound is not None:
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
            working = working.crop(bbox)

    pcd_down = working.voxel_down_sample(voxel_size)
    normal_config = dict(config.get("normals", {}))
    normal_radius = voxel_size * float(normal_config.get("radius_factor", 2.0))
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=int(normal_config.get("max_nn", 30)),
        )
    )

    fpfh_config = dict(config.get("fpfh", {}))
    fpfh_radius = voxel_size * float(fpfh_config.get("radius_factor", 5.0))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_radius,
            max_nn=int(fpfh_config.get("max_nn", 100)),
        ),
    )
    return {"pcd": working, "pcd_down": pcd_down, "fpfh": fpfh, "voxel_size": voxel_size}

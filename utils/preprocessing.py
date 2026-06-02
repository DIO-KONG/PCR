from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from utils import cache, io


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


def prepare_point_cloud_from_path_with_cache(path: str | Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(config or {})
    raw_pcd = io.read_point_cloud(path)
    cache_config = dict(config.get("cache", {}))
    cache_enabled = bool(cache_config.get("enabled", True))
    cache_dir = Path(cache_config.get("dir", "data/cache"))
    key = cache.cache_key(path, config)
    prepared_cache_path = cache.cache_path(cache_dir, key)

    cache_error = None
    if cache_enabled and cache.is_cache_hit(prepared_cache_path):
        try:
            prepared = cache.load_cache(prepared_cache_path)
            prepared["cache_hit"] = True
        except Exception as exc:
            cache_error = f"cache load failed: {exc}"
            prepared = prepare_point_cloud(raw_pcd, config)
            prepared["cache_hit"] = False
    else:
        prepared = prepare_point_cloud(raw_pcd, config)
        cache_payload = {k: v for k, v in prepared.items() if k != "pcd"}
        if cache_enabled:
            try:
                cache.save_cache(prepared_cache_path, cache_payload)
            except Exception as exc:
                cache_error = f"cache save failed: {exc}"
        prepared["cache_hit"] = False

    prepared["pcd"] = raw_pcd
    prepared["raw_pcd"] = raw_pcd
    prepared["cache_key"] = key
    prepared["cache_path"] = str(prepared_cache_path)
    if cache_error:
        prepared["cache_error"] = cache_error
    return prepared

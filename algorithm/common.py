from __future__ import annotations

import numpy as np
import open3d as o3d

from algorithm.base import RegistrationCandidate
from utils.metrics import evaluate_registration, gate_status, matrix_diagnostics, strict_score
from utils.pointcloud import transform_cloud
from utils.preprocess import estimate_normals, preprocess_for_registration


def prepare_clouds(source, target, config: dict):
    """按算法配置预处理 source/target。

    默认参数略激进：先做地板对齐和去地板，再降采样到 8cm。
    这样后续 fitness 不容易因为地板大平面或过密点云而虚高。
    """

    preprocess = config.get("preprocess", {})
    kwargs = {
        "align_floor": bool(preprocess.get("align_floor", True)),
        "remove_floor": bool(preprocess.get("remove_floor", True)),
        "floor_distance_threshold": float(preprocess.get("floor_distance_threshold", 0.03)),
        "voxel_size": float(preprocess.get("voxel_size", 0.08)),
        "sor": bool(preprocess.get("sor", True)),
        "sor_nb_neighbors": int(preprocess.get("sor_nb_neighbors", 20)),
        "sor_std_ratio": float(preprocess.get("sor_std_ratio", 1.5)),
        "normals": bool(preprocess.get("estimate_normals", True)),
        "normal_radius": float(preprocess.get("normal_radius", 0.20)),
        "normal_max_nn": int(preprocess.get("normal_max_nn", 30)),
        "floor_kwargs": preprocess.get("floor", {}),
    }
    source_prepared = preprocess_for_registration(source, **kwargs)
    target_prepared = preprocess_for_registration(target, **kwargs)
    return source_prepared.cloud, target_prepared.cloud


def compute_fpfh(cloud: o3d.geometry.PointCloud, config: dict) -> o3d.pipelines.registration.Feature:
    """计算 FPFH 特征。"""

    fpfh_radius = float(config.get("fpfh_radius", 0.35))
    fpfh_max_nn = int(config.get("fpfh_max_nn", 100))
    return o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fpfh_radius, max_nn=fpfh_max_nn),
    )


def ensure_normals(cloud: o3d.geometry.PointCloud, radius: float = 0.20, max_nn: int = 30):
    """保证点云有法线。"""

    if cloud.has_normals():
        return cloud
    return estimate_normals(cloud, radius=radius, max_nn=max_nn)


def evaluate_matrix(
    candidate_id: int,
    matrix: np.ndarray,
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    config: dict,
    *,
    metadata: dict | None = None,
) -> RegistrationCandidate:
    """应用矩阵并计算严格指标。"""

    evaluation = config.get("evaluation", {})
    transformed = transform_cloud(source, matrix)
    metrics = {
        **evaluate_registration(
            transformed,
            target,
            threshold=float(evaluation.get("threshold", 0.08)),
            trimmed_ratio=float(evaluation.get("trimmed_ratio", 0.8)),
            max_eval_points=int(evaluation.get("max_eval_points", 30000)),
        ),
        **matrix_diagnostics(matrix),
    }
    metrics["status"] = gate_status(metrics, config.get("gate", {}))
    return RegistrationCandidate(
        candidate_id=candidate_id,
        matrix=matrix,
        score=strict_score(metrics),
        metrics=metrics,
        metadata=metadata or {},
    )


def select_best(candidates: list[RegistrationCandidate]) -> RegistrationCandidate:
    """优先选过严格 gate 的最高分候选，否则选最高分候选。"""

    accepted = [item for item in candidates if item.metrics.get("status") == "accepted_by_strict_gate"]
    pool = accepted if accepted else candidates
    return sorted(pool, key=lambda item: item.score, reverse=True)[0]


def run_point_to_plane_icp(source, target, init: np.ndarray, config: dict):
    """Open3D point-to-plane ICP。"""

    icp_cfg = config.get("icp", {})
    source = ensure_normals(source)
    target = ensure_normals(target)
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(icp_cfg.get("max_correspondence_distance", 0.08)),
        init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(icp_cfg.get("max_iterations", 40))
        ),
    )


def run_colored_icp(source, target, init: np.ndarray, config: dict):
    """Open3D Colored ICP。"""

    colored_cfg = config.get("colored_icp", {})
    source = ensure_normals(source)
    target = ensure_normals(target)
    return o3d.pipelines.registration.registration_colored_icp(
        source,
        target,
        float(colored_cfg.get("max_correspondence_distance", 0.08)),
        init,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(colored_cfg.get("max_iterations", 30))
        ),
    )

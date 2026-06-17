from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from algorithm.base import RegistrationResult
from algorithm.common import compute_fpfh, evaluate_matrix, prepare_clouds, run_point_to_plane_icp, select_best
from utils.pointcloud import load_cloud


def feature_correspondences(source_fpfh, target_fpfh, max_correspondences: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """用 FPFH 特征最近邻构造 TEASER 对应点索引。

    这里用互最近邻过滤，尽量减少明显错误对应。TEASER 自身会处理外点，
    但输入对应关系太脏时仍会降低成功率。
    """

    source_feat = np.asarray(source_fpfh.data).T
    target_feat = np.asarray(target_fpfh.data).T
    target_tree = o3d.geometry.KDTreeFlann(target_feat.T)
    source_to_target = []
    for source_index, feat in enumerate(source_feat):
        count, indices, _ = target_tree.search_knn_vector_xd(feat, 1)
        if count:
            source_to_target.append((source_index, int(indices[0])))

    source_tree = o3d.geometry.KDTreeFlann(source_feat.T)
    mutual = []
    for source_index, target_index in source_to_target:
        count, indices, _ = source_tree.search_knn_vector_xd(target_feat[target_index], 1)
        if count and int(indices[0]) == source_index:
            mutual.append((source_index, target_index))

    if len(mutual) > max_correspondences:
        select = np.linspace(0, len(mutual) - 1, max_correspondences).astype(int)
        mutual = [mutual[index] for index in select]
    if not mutual:
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    pairs = np.asarray(mutual, dtype=int)
    return pairs[:, 0], pairs[:, 1]


def run_teaser(source_down, target_down, config: dict) -> np.ndarray:
    """运行 TEASER++ 求刚体粗配准矩阵。"""

    try:
        import teaserpp_python
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "teaserpp_python is not installed in .env. Install TEASER++ Python binding to enable this scheme."
        ) from exc

    source_fpfh = compute_fpfh(source_down, config.get("features", {}))
    target_fpfh = compute_fpfh(target_down, config.get("features", {}))
    source_idx, target_idx = feature_correspondences(
        source_fpfh,
        target_fpfh,
        max_correspondences=int(config.get("teaser", {}).get("max_correspondences", 5000)),
    )
    if len(source_idx) < 6:
        raise RuntimeError(f"Not enough mutual FPFH correspondences for TEASER: {len(source_idx)}")

    source_points = np.asarray(source_down.points)[source_idx].T
    target_points = np.asarray(target_down.points)[target_idx].T

    teaser_cfg = config.get("teaser", {})
    params = teaserpp_python.RobustRegistrationSolver.Params()
    params.cbar2 = float(teaser_cfg.get("cbar2", 1.0))
    params.noise_bound = float(teaser_cfg.get("noise_bound", 0.08))
    params.estimate_scaling = False
    params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    params.rotation_gnc_factor = float(teaser_cfg.get("rotation_gnc_factor", 1.4))
    params.rotation_max_iterations = int(teaser_cfg.get("rotation_max_iterations", 100))
    params.rotation_cost_threshold = float(teaser_cfg.get("rotation_cost_threshold", 1e-12))

    solver = teaserpp_python.RobustRegistrationSolver(params)
    solver.solve(source_points, target_points)
    solution = solver.getSolution()

    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(solution.rotation, dtype=float)
    matrix[:3, 3] = np.asarray(solution.translation, dtype=float)
    return matrix


def register(source_path: str | Path, target_path: str | Path, config: dict) -> RegistrationResult:
    """TEASER++ + point-to-plane ICP 方案。"""

    source_path = Path(source_path)
    target_path = Path(target_path)
    source = load_cloud(source_path)
    target = load_cloud(target_path)
    source_down, target_down = prepare_clouds(source, target, config)

    try:
        coarse_matrix = run_teaser(source_down, target_down, config)
    except RuntimeError as exc:
        identity = np.eye(4)
        candidate = evaluate_matrix(
            1,
            identity,
            source_down,
            target_down,
            config,
            metadata={"dependency_error": str(exc), "coarse_method": "teaser"},
        )
        return RegistrationResult(
            algorithm="teaser_point_to_plane_icp",
            source_path=source_path,
            target_path=target_path,
            status="skipped_missing_teaser_dependency",
            matrix=identity,
            metrics=candidate.metrics,
            candidates=[candidate],
            message=str(exc),
        )

    coarse_candidate = evaluate_matrix(
        1,
        coarse_matrix,
        source_down,
        target_down,
        config,
        metadata={"coarse_method": "teaser"},
    )
    icp = run_point_to_plane_icp(source_down, target_down, coarse_candidate.matrix, config)
    refined = evaluate_matrix(
        2,
        np.asarray(icp.transformation, dtype=float),
        source_down,
        target_down,
        config,
        metadata={
            "coarse_method": "teaser",
            "refine_method": "point_to_plane_icp",
            "icp_fitness": float(icp.fitness),
            "icp_inlier_rmse": float(icp.inlier_rmse),
        },
    )
    candidates = sorted([coarse_candidate, refined], key=lambda item: item.score, reverse=True)
    best = select_best(candidates)
    return RegistrationResult(
        algorithm="teaser_point_to_plane_icp",
        source_path=source_path,
        target_path=target_path,
        status=str(best.metrics["status"]),
        matrix=best.matrix,
        metrics=best.metrics,
        candidates=candidates,
    )

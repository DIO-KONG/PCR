from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithm.base import RegistrationResult
from algorithm.common import evaluate_matrix, prepare_clouds, run_point_to_plane_icp, select_best
from algorithm.fpfh_ransac import register as run_ransac
from utils.pointcloud import load_cloud


def register(source_path: str | Path, target_path: str | Path, config: dict) -> RegistrationResult:
    """FPFH + RANSAC + point-to-plane ICP。

    RANSAC 负责粗配准，ICP 只做小范围几何精修。ICP correspondence distance 默认 8cm，
    故意不放宽，避免被相似隔板或远距离错误对应拉偏。
    """

    source_path = Path(source_path)
    target_path = Path(target_path)
    coarse = run_ransac(source_path, target_path, config)
    source = load_cloud(source_path)
    target = load_cloud(target_path)
    source_down, target_down = prepare_clouds(source, target, config)

    candidates = []
    for index, candidate in enumerate(coarse.candidates, 1):
        icp = run_point_to_plane_icp(source_down, target_down, candidate.matrix, config)
        matrix = np.asarray(icp.transformation, dtype=float)
        refined = evaluate_matrix(
            index,
            matrix,
            source_down,
            target_down,
            config,
            metadata={
                **candidate.metadata,
                "refine_method": "point_to_plane_icp",
                "icp_fitness": float(icp.fitness),
                "icp_inlier_rmse": float(icp.inlier_rmse),
            },
        )
        candidates.append(refined)

    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    best = select_best(candidates)
    return RegistrationResult(
        algorithm="fpfh_ransac_point_to_plane_icp",
        source_path=source_path,
        target_path=target_path,
        status=str(best.metrics["status"]),
        matrix=best.matrix,
        metrics=best.metrics,
        candidates=candidates[: int(config.get("output", {}).get("top_k", 5))],
    )

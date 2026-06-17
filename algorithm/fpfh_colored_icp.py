from __future__ import annotations

from pathlib import Path

import numpy as np

from algorithm.base import RegistrationResult
from algorithm.common import evaluate_matrix, prepare_clouds, run_colored_icp, select_best
from algorithm.fpfh_ransac import register as run_ransac
from utils.pointcloud import load_cloud


def register(source_path: str | Path, target_path: str | Path, config: dict) -> RegistrationResult:
    """FPFH + RANSAC + Colored ICP。

    Colored ICP 使用颜色和几何共同约束，适合办公室重复隔板结构中颜色差异有帮助的情况。
    仍然保持较小 correspondence distance，防止颜色相似区域被远距离拉走。
    """

    source_path = Path(source_path)
    target_path = Path(target_path)
    coarse = run_ransac(source_path, target_path, config)
    source = load_cloud(source_path)
    target = load_cloud(target_path)
    source_down, target_down = prepare_clouds(source, target, config)

    candidates = []
    for index, candidate in enumerate(coarse.candidates, 1):
        colored = run_colored_icp(source_down, target_down, candidate.matrix, config)
        matrix = np.asarray(colored.transformation, dtype=float)
        refined = evaluate_matrix(
            index,
            matrix,
            source_down,
            target_down,
            config,
            metadata={
                **candidate.metadata,
                "refine_method": "colored_icp",
                "colored_fitness": float(colored.fitness),
                "colored_inlier_rmse": float(colored.inlier_rmse),
            },
        )
        candidates.append(refined)

    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    best = select_best(candidates)
    return RegistrationResult(
        algorithm="fpfh_ransac_colored_icp",
        source_path=source_path,
        target_path=target_path,
        status=str(best.metrics["status"]),
        matrix=best.matrix,
        metrics=best.metrics,
        candidates=candidates[: int(config.get("output", {}).get("top_k", 5))],
    )

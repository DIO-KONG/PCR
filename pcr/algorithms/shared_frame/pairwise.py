from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pcr.algorithms.refinement.icp import BoundedPointToPlaneIcp
from pcr.algorithms.shared_frame.coarse import ransac_rigid_candidates
from pcr.algorithms.shared_frame.correspondences import collect_correspondences
from pcr.domain import EvaluationMetrics, PairwiseRegistrationResult, Transform
from pcr.evaluation.ranking import score_metrics
from pcr.evaluation.registration_metrics import evaluate_shared_correspondences
from pcr.io.da3_npz import load_da3_batch
from pcr.io.pointcloud_io import load_point_cloud


def register_shared_frame_pairwise(
    source_path: str | Path,
    target_path: str | Path,
    config: dict[str, Any],
) -> PairwiseRegistrationResult:
    """执行一次 shared-frame source->target 刚体配准。

    这是旧 shared-frame pairwise register 入口的新位置。它只做数值计算和
    结构化返回，不写 artifact，不依赖 testbench。
    """

    source_npz = Path(config["source_npz"])
    target_npz = Path(config["target_npz"])
    source_batch = load_da3_batch(source_npz)
    target_batch = load_da3_batch(target_npz)
    source_points, target_points, frame_names, frame_stats = collect_correspondences(source_batch, target_batch, config)

    candidates = ransac_rigid_candidates(
        source_points,
        target_points,
        frame_names,
        config,
        source_frame="source",
        target_frame="target",
    )
    top_k = int(config.get("output", {}).get("top_k", len(candidates)))
    candidates = candidates[:top_k]
    best = candidates[0]
    evaluation_threshold = float(config.get("ransac", {}).get("evaluation_threshold", config.get("ransac", {}).get("inlier_threshold", 0.06)))
    pre_icp_metrics = best.metrics.to_dict()
    final_transform = best.transform
    icp_metrics: dict[str, Any] = {"enabled": False, "status": "disabled"}

    icp_config = config.get("icp_refinement", {})
    if bool(icp_config.get("enabled", False)):
        source_cloud_path = Path(config.get("source_cloud", source_path))
        target_cloud_path = Path(config.get("target_cloud", target_path))
        source_cloud = load_point_cloud(source_cloud_path)
        target_cloud = load_point_cloud(target_cloud_path)
        refinement = BoundedPointToPlaneIcp().refine(
            source_cloud=source_cloud,
            target_world=target_cloud,
            initial_transform=best.transform,
            icp_config=icp_config,
        )
        final_transform = refinement.accepted_transform
        icp_metrics = {"enabled": True, **refinement.metrics.to_dict()}

        # 办公室重复结构中 ICP 可能提高点云 fitness，却破坏共享帧几何一致性。
        # pairwise 实验保留这道兜底；walk-forward 版本当前只把共享帧一致性作为诊断。
        shared_after_icp = evaluate_shared_correspondences(
            source_points,
            target_points,
            frame_names,
            final_transform.matrix,
            evaluation_threshold,
        )
        if refinement.accepted:
            max_median_ratio = float(icp_config.get("max_shared_median_error_ratio", 1.10))
            min_inlier_ratio_ratio = float(icp_config.get("min_shared_inlier_ratio_ratio", 0.85))
            pre_median = float(pre_icp_metrics.get("median_error", float("inf")))
            pre_inlier_ratio = float(pre_icp_metrics.get("inlier_ratio", 0.0))
            median_ok = shared_after_icp["median_error"] <= pre_median * max_median_ratio
            inlier_ok = shared_after_icp["inlier_ratio"] >= pre_inlier_ratio * min_inlier_ratio_ratio
            if not (median_ok and inlier_ok):
                icp_metrics = {
                    **icp_metrics,
                    "status": "rejected_by_shared_consistency",
                    "shared_median_error_after_icp": shared_after_icp["median_error"],
                    "shared_median_error_before_icp": pre_median,
                    "shared_inlier_ratio_after_icp": shared_after_icp["inlier_ratio"],
                    "shared_inlier_ratio_before_icp": pre_inlier_ratio,
                    "max_shared_median_error_ratio": max_median_ratio,
                    "min_shared_inlier_ratio_ratio": min_inlier_ratio_ratio,
                }
                final_transform = best.transform

    metrics = evaluate_shared_correspondences(
        source_points,
        target_points,
        frame_names,
        final_transform.matrix,
        evaluation_threshold,
    )
    metrics.update(
        {
            "coarse_threshold": pre_icp_metrics.get("coarse_threshold"),
            "evaluation_threshold": evaluation_threshold,
            "ransac_iterations": pre_icp_metrics.get("ransac_iterations"),
            "ransac_sample_size": pre_icp_metrics.get("ransac_sample_size"),
            "refit_inlier_count": pre_icp_metrics.get("refit_inlier_count"),
            "refinement_history": pre_icp_metrics.get("refinement_history", []),
            "pre_icp_shared_metrics": pre_icp_metrics,
            "icp_refinement": icp_metrics,
            "correspondence_count": int(len(source_points)),
            "shared_frames": frame_stats,
            "allow_scale": False,
            "source_npz": str(source_npz),
            "target_npz": str(target_npz),
        }
    )
    status = "accepted" if metrics["inlier_ratio"] >= float(config.get("ransac", {}).get("min_inlier_ratio", 0.35)) else "needs_review_low_inlier_ratio"
    metrics["status"] = status

    final_candidate = best
    final_candidate = type(best)(
        candidate_id=best.candidate_id,
        transform=final_transform,
        score=score_metrics(metrics),
        metrics=EvaluationMetrics.from_mapping(metrics),
        metadata=best.metadata,
    )
    candidates = (final_candidate, *tuple(candidates[1:]))
    return PairwiseRegistrationResult(
        algorithm="shared_frame_alignment",
        source_path=Path(source_path),
        target_path=Path(target_path),
        status=status,
        transform=final_transform,
        metrics=metrics,
        candidates=tuple(candidates),
    )

from __future__ import annotations

from typing import Any

from pcr.domain import FusionReport, QualityReport, QualityStatus, StepResult


def evaluate_pose_quality(step: StepResult, config: dict[str, Any]) -> QualityReport:
    """根据共享帧与 ICP 指标判断当前位姿是否可信。

    第一版门控保持可解释的硬阈值，不把质量决策藏在复杂分数里。位姿即使
    `needs_review` 也会保留 transform，方便离线排查；只有 `rejected` 会阻止融合。
    """

    shared = step.coarse_shared_metrics
    icp = step.refinement.metrics
    reasons: list[str] = []
    metrics = {
        "shared_median_error": shared.median_error,
        "shared_p90_error": shared.p90_error,
        "icp_fitness": icp.fitness,
        "icp_rmse": icp.inlier_rmse,
        "icp_translation_delta": icp.translation_delta,
        "icp_rotation_delta_deg": icp.rotation_delta_deg,
        "icp_status": icp.status,
    }

    if shared.median_error > float(config.get("max_shared_median_error", 0.06)):
        reasons.append("shared_median_error_too_high")
    if shared.p90_error > float(config.get("max_shared_p90_error", 0.12)):
        reasons.append("shared_p90_error_too_high")
    if icp.inlier_rmse > float(config.get("max_icp_rmse", 0.08)):
        reasons.append("icp_rmse_too_high")
    if icp.fitness < float(config.get("min_icp_fitness", 0.35)):
        reasons.append("icp_fitness_too_low")
    if icp.translation_delta > float(config.get("max_icp_translation_delta", 0.15)):
        reasons.append("icp_translation_delta_too_large")
    if icp.rotation_delta_deg > float(config.get("max_icp_rotation_delta_deg", 5.0)):
        reasons.append("icp_rotation_delta_too_large")
    if step.refinement.accepted is False:
        reasons.append("icp_rejected_by_boundary")

    reject_reasons = {"icp_translation_delta_too_large", "icp_rotation_delta_too_large"}
    if any(reason in reject_reasons for reason in reasons):
        pose_status = QualityStatus.REJECTED
        fusion_status = QualityStatus.REJECTED
    elif reasons:
        pose_status = QualityStatus.NEEDS_REVIEW
        fusion_status = QualityStatus.NEEDS_REVIEW
    else:
        pose_status = QualityStatus.ACCEPTED
        fusion_status = QualityStatus.ACCEPTED

    return QualityReport(
        pose_status=pose_status,
        fusion_status=fusion_status,
        reasons=tuple(reasons),
        metrics=metrics,
    )


def apply_fusion_quality(
    pose_report: QualityReport,
    fusion_report: FusionReport,
    config: dict[str, Any],
) -> QualityReport:
    """把融合冲突率纳入质量报告。"""

    reasons = list(pose_report.reasons)
    metrics = dict(pose_report.metrics)
    metrics.update(
        {
            "fusion_conflict_ratio": fusion_report.conflict_ratio,
            "fusion_accepted_points": fusion_report.accepted_points,
            "fusion_duplicate_points": fusion_report.duplicate_points,
            "fusion_conflict_points": fusion_report.conflict_points,
            "fusion_input_points": fusion_report.input_points,
        }
    )
    max_conflict_ratio = float(config.get("max_conflict_ratio", 0.30))
    if fusion_report.conflict_ratio > max_conflict_ratio:
        reasons.append("fusion_conflict_ratio_too_high")

    if pose_report.pose_status == QualityStatus.REJECTED:
        fusion_status = QualityStatus.REJECTED
    elif fusion_report.conflict_ratio > max_conflict_ratio:
        fusion_status = QualityStatus.REJECTED
    elif reasons:
        fusion_status = QualityStatus.NEEDS_REVIEW
    else:
        fusion_status = QualityStatus.ACCEPTED

    return QualityReport(
        pose_status=pose_report.pose_status,
        fusion_status=fusion_status,
        reasons=tuple(reasons),
        metrics=metrics,
    )

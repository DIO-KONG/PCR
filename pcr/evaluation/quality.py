from __future__ import annotations

from typing import Any

from pcr.domain import FusionReport, QualityReport, QualityStatus, StepResult


def config_bool(config: dict[str, Any], key: str, default: bool = False) -> bool:
    """从 YAML 配置读取布尔值，避免字符串 `"false"` 被当作 True。"""

    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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
    fusion_params = dict(fusion_report.params)
    metrics.update(
        {
            "fusion_conflict_ratio": fusion_report.conflict_ratio,
            "fusion_duplicate_ratio": fusion_report.duplicate_ratio,
            "fusion_accepted_ratio": fusion_report.accepted_ratio,
            "fusion_accepted_points": fusion_report.accepted_points,
            "fusion_duplicate_points": fusion_report.duplicate_points,
            "fusion_conflict_points": fusion_report.conflict_points,
            "fusion_input_points": fusion_report.input_points,
            "fusion_accepted_xz_area": fusion_params.get("accepted_xz_area", 0.0),
            "fusion_accepted_extent_max": fusion_params.get("accepted_extent_max", 0.0),
        }
    )
    max_conflict_ratio = float(config.get("max_conflict_ratio", 0.30))
    if fusion_report.conflict_ratio > max_conflict_ratio:
        reasons.append("fusion_conflict_ratio_too_high")

    scale_gate_enabled = config_bool(config, "enable_scale_shadow_gate", False)
    severe_scale_shadow = (
        fusion_report.conflict_ratio > float(config.get("scale_shadow_min_conflict_ratio", 0.55))
        and fusion_report.accepted_ratio > float(config.get("scale_shadow_severe_min_accepted_ratio", 0.05))
        and metrics["icp_fitness"] < float(config.get("scale_shadow_severe_max_icp_fitness", 0.55))
        and float(metrics["fusion_accepted_xz_area"]) > float(config.get("scale_shadow_severe_min_xz_area", 25.0))
        and float(metrics["fusion_accepted_extent_max"]) > float(config.get("scale_shadow_severe_min_extent", 7.0))
    )
    early_sparse_scale_shadow = (
        fusion_report.conflict_ratio > float(config.get("scale_shadow_min_conflict_ratio", 0.55))
        and float(config.get("scale_shadow_early_min_accepted_ratio", 0.015)) < fusion_report.accepted_ratio < float(config.get("scale_shadow_early_max_accepted_ratio", 0.05))
        and fusion_report.duplicate_ratio < float(config.get("scale_shadow_early_max_duplicate_ratio", 0.45))
        and metrics["icp_fitness"] < float(config.get("scale_shadow_early_max_icp_fitness", 0.65))
        and metrics["shared_p90_error"] < float(config.get("scale_shadow_early_max_shared_p90_error", 0.10))
        and float(metrics["fusion_accepted_xz_area"]) > float(config.get("scale_shadow_early_min_xz_area", 15.0))
        and float(metrics["fusion_accepted_extent_max"]) > float(config.get("scale_shadow_early_min_extent", 5.0))
    )
    metrics["fusion_severe_scale_shadow"] = severe_scale_shadow
    metrics["fusion_early_sparse_scale_shadow"] = early_sparse_scale_shadow
    scale_shadow_quarantined = scale_gate_enabled and (severe_scale_shadow or early_sparse_scale_shadow)
    if scale_shadow_quarantined:
        reasons.append("fusion_scale_shadow_quarantined")

    if pose_report.pose_status == QualityStatus.REJECTED:
        fusion_status = QualityStatus.REJECTED
    elif scale_shadow_quarantined:
        fusion_status = QualityStatus.REJECTED
    elif fusion_report.conflict_ratio > max_conflict_ratio:
        # fusion 采用点级过滤：accepted 写入、duplicate 更新、conflict 丢弃。
        # 因此高冲突率只表示需要人工复查，不再回滚整步融合结果。
        fusion_status = QualityStatus.NEEDS_REVIEW
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

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from .transforms import Transform


class PointCloudRole(str, Enum):
    """点云在流程中的角色。

    用枚举避免不同模块随手写 `"raw"`、`"floor_removed"` 等字符串后产生
    隐式分叉。
    """

    RAW = "raw"
    PREPROCESSED = "preprocessed"
    REGISTERED = "registered"
    WORLD = "world"
    OVERLAY = "overlay"
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"


class QualityStatus(str, Enum):
    """质量门控状态。"""

    ACCEPTED = "accepted"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


@dataclass(frozen=True)
class PointCloudRef:
    """磁盘点云引用，只描述位置和语义，不持有 Open3D 对象。"""

    path: Path
    cloud_id: str
    role: PointCloudRole


@dataclass(frozen=True)
class BatchRef:
    """一个 DA3 batch 的所有关键输入引用。"""

    batch_id: str
    npz_path: Path
    raw_cloud_path: Path
    preprocessed_cloud_path: Path


@dataclass(frozen=True)
class QualityReport:
    """单步配准与融合质量门控结果。"""

    pose_status: QualityStatus
    fusion_status: QualityStatus
    reasons: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def pose_accepted(self) -> bool:
        return self.pose_status != QualityStatus.REJECTED

    @property
    def fusion_accepted(self) -> bool:
        return self.fusion_status == QualityStatus.ACCEPTED


@dataclass(frozen=True)
class FusionReport:
    """一次保守融合的点分类统计。"""

    accepted_points: int
    duplicate_points: int
    conflict_points: int
    rejected_points: int
    input_points: int
    conflict_ratio: float
    duplicate_ratio: float
    accepted_ratio: float
    new_frame_names: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusionDebugClouds:
    """保守融合为了人工检查输出的三类点云。"""

    accepted: o3d.geometry.PointCloud
    duplicate: o3d.geometry.PointCloud
    conflict: o3d.geometry.PointCloud


@dataclass(frozen=True)
class FusionResult:
    """融合策略输出，不负责写 artifact。"""

    fused_cloud: o3d.geometry.PointCloud
    report: FusionReport
    debug_clouds: FusionDebugClouds


@dataclass(frozen=True)
class SubmapEdge:
    """相邻 submap 的位姿链边。"""

    source_submap_id: str
    target_submap_id: str
    transform: Transform
    status: str
    reason: str = ""


@dataclass(frozen=True)
class RegistrationTask:
    """单步 source batch 到 target batch 的配准任务。"""

    step_id: str
    step_index: int
    source: BatchRef
    target: BatchRef
    params: dict[str, Any]


@dataclass(frozen=True)
class SequenceTask:
    """完整 walk-forward 任务。"""

    name: str
    baseline: BatchRef
    steps: tuple[RegistrationTask, ...]
    params: dict[str, Any]


@dataclass(frozen=True)
class EvaluationMetrics:
    """共享帧或 ICP 的指标容器。

    `extra` 用来承接算法探索阶段仍在变化的字段，但稳定字段被显式列出，
    避免所有模块都读写大型嵌套 dict。
    """

    inlier_ratio: float = 0.0
    inlier_count: int = 0
    rmse: float = float("inf")
    median_error: float = float("inf")
    p90_error: float = float("inf")
    mean_error: float = float("inf")
    threshold: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "EvaluationMetrics":
        """从旧算法 dict 指标转换到结构化指标。"""

        return cls(
            inlier_ratio=float(payload.get("inlier_ratio", 0.0)),
            inlier_count=int(payload.get("inlier_count", 0)),
            rmse=float(payload.get("rmse", payload.get("inlier_rmse", float("inf")))),
            median_error=float(payload.get("median_error", float("inf"))),
            p90_error=float(payload.get("p90_error", float("inf"))),
            mean_error=float(payload.get("mean_error", float("inf"))),
            threshold=float(payload.get("inlier_threshold", payload.get("evaluation_threshold", 0.0))),
            extra=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        """输出兼容旧 artifact 的 dict。"""

        payload = dict(self.extra)
        payload.update(
            {
                "inlier_ratio": self.inlier_ratio,
                "inlier_count": self.inlier_count,
                "rmse": self.rmse,
                "median_error": self.median_error,
                "p90_error": self.p90_error,
                "mean_error": self.mean_error,
                "inlier_threshold": self.threshold,
            }
        )
        return payload


@dataclass(frozen=True)
class FrameSelectionCandidate:
    """一组共享帧组合及其粗配准评分。"""

    frames: tuple[str, ...]
    score: float
    correspondence_count: int
    metrics: EvaluationMetrics
    raw_row: dict[str, Any]


@dataclass(frozen=True)
class FrameSelection:
    """动态共享帧选择结果。"""

    shared_frames: tuple[str, ...]
    selected_frames: tuple[str, ...]
    ranking: tuple[FrameSelectionCandidate, ...]


@dataclass(frozen=True)
class RegistrationCandidate:
    """一个 coarse registration 候选。"""

    candidate_id: str
    transform: Transform
    score: float
    metrics: EvaluationMetrics
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistrationResult:
    """coarse registration 的结构化输出。"""

    selected: RegistrationCandidate
    candidates: tuple[RegistrationCandidate, ...]
    frame_selection: FrameSelection
    source_points: np.ndarray
    target_points: np.ndarray
    frame_names: np.ndarray
    status: str = "accepted"


@dataclass(frozen=True)
class IcpMetrics:
    """bounded ICP 的主要指标。"""

    status: str
    fitness: float
    inlier_rmse: float
    translation_delta: float
    rotation_delta_deg: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra)
        payload.update(
            {
                "status": self.status,
                "fitness": self.fitness,
                "inlier_rmse": self.inlier_rmse,
                "translation_delta": self.translation_delta,
                "rotation_delta_deg": self.rotation_delta_deg,
            }
        )
        return payload


@dataclass(frozen=True)
class RefinementResult:
    """ICP refine 输出。

    `refined_transform` 是 ICP 实际求出的矩阵；`accepted_transform` 是经过边界
    判断后真正进入后续流程的矩阵。
    """

    initial_transform: Transform
    refined_transform: Transform
    delta_from_initial: Transform
    accepted_transform: Transform
    accepted: bool
    metrics: IcpMetrics


@dataclass(frozen=True)
class StepResult:
    """单步配准结果，不负责保存文件。"""

    task: RegistrationTask
    coarse: RegistrationResult
    refinement: RefinementResult
    final_transform: Transform
    final_source: str
    final_registered_source: o3d.geometry.PointCloud
    coarse_registered_source: o3d.geometry.PointCloud
    refined_registered_source: o3d.geometry.PointCloud
    coarse_shared_metrics: EvaluationMetrics
    source_cloud_points: int


@dataclass(frozen=True)
class WorldUpdateResult:
    """一次 StepResult 应用到 WorldState 后的结果。"""

    step_result: StepResult
    fused_world: o3d.geometry.PointCloud
    fused_world_points: int
    world_stats: dict[str, Any]


@dataclass(frozen=True)
class SubmapStepResult:
    """submap pipeline 中单步配准、质量门控和融合的完整结果。"""

    step_result: StepResult
    quality: QualityReport
    fusion: FusionResult | None
    active_submap_id: str
    world_before_step: o3d.geometry.PointCloud
    frame_combo_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class SubmapPipelineResult:
    """完整 online submap sequence 的结果。"""

    run_name: str
    run_dir: Path
    steps: tuple[SubmapStepResult, ...]
    submap_edges: tuple[SubmapEdge, ...]
    final_submap_id: str
    global_preview_points: int


@dataclass
class RegistrationContext:
    """pipeline 运行时上下文。"""

    world: Any
    run_name: str


@dataclass(frozen=True)
class PipelineResult:
    """完整 sequence 结束后的结构化结果。"""

    run_name: str
    run_dir: Path
    steps: tuple[WorldUpdateResult, ...]
    final_world_points: int


@dataclass(frozen=True)
class PairwiseRegistrationResult:
    """pairwise registration 入口的统一输出。

    它面向 `run_registration.py` 这类单任务实验：只需要一个最终
    source->target 变换、候选列表和指标，不涉及 WorldState。
    """

    algorithm: str
    source_path: Path
    target_path: Path
    status: str
    transform: Transform
    metrics: dict[str, Any]
    candidates: tuple[RegistrationCandidate, ...] = ()
    message: str = ""

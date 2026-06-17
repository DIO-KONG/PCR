from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from pcr.algorithms.shared_frame.coarse import ransac_rigid_candidates
from pcr.algorithms.shared_frame.correspondences import collect_correspondences
from pcr.domain import (
    EvaluationMetrics,
    FrameSelection,
    FrameSelectionCandidate,
    RegistrationCandidate,
    RegistrationResult,
    Transform,
)
from pcr.evaluation.registration_metrics import evaluate_shared_correspondences
from pcr.io.da3_npz import image_names, load_da3_batch


def frame_sort_key(name: str) -> tuple[int, str]:
    """按文件名前缀数字排序，保证 `10-rgb.png` 排在 `9-rgb.png` 后。"""

    match = re.match(r"^(\d+)", str(name))
    if match:
        return int(match.group(1)), str(name)
    return 10**9, str(name)


def discover_shared_frames(source_batch: dict[str, np.ndarray], target_batch: dict[str, np.ndarray]) -> list[str]:
    """自动发现两个 DA3 batch 的共享 RGB 帧。"""

    source_names = set(image_names(source_batch))
    target_names = set(image_names(target_batch))
    return sorted(source_names & target_names, key=frame_sort_key)


def make_shared_frame_config(
    algorithm_params: dict[str, Any],
    source_npz: str,
    target_npz: str,
    shared_frames: list[str] | None = None,
) -> dict[str, Any]:
    """构造 shared-frame 数值组件需要的配置。"""

    config = {
        "source_npz": source_npz,
        "target_npz": target_npz,
        "sampling": dict(algorithm_params["sampling"]),
        "ransac": dict(algorithm_params["ransac"]),
        "refinement": dict(algorithm_params["refinement"]),
        "icp_refinement": {"enabled": False},
    }
    if shared_frames is not None:
        config["shared_frames"] = list(shared_frames)
    return config


@dataclass(frozen=True)
class FrameSelectionPayload:
    """frame selection 的完整输出。

    `registration` 是结构化结果；后面的点数组保留给 step pipeline 计算
    shared-frame 诊断指标，避免再次采样。
    """

    registration: RegistrationResult
    rows: list[dict[str, Any]]


class DynamicTopKSharedFrameSelector:
    """动态共享帧 4选3 粗配准选择器。

    当前默认用于“相邻 DA3 batch 共享 4 帧，动态选表现最好的 3 帧”的场景。
    它只返回候选和指标，不写文件、不更新 world。
    """

    def select(
        self,
        *,
        source_batch_id: str,
        target_batch_id: str,
        source_npz: str,
        target_npz: str,
        algorithm_params: dict[str, Any],
    ) -> FrameSelectionPayload:
        source_batch = load_da3_batch(source_npz)
        target_batch = load_da3_batch(target_npz)
        shared_frames = discover_shared_frames(source_batch, target_batch)
        combo_size = int(algorithm_params.get("frame_combo_size", 3))
        if len(shared_frames) < combo_size:
            raise RuntimeError(f"Not enough shared frames: shared={shared_frames}, combo_size={combo_size}")

        rows: list[dict[str, Any]] = []
        candidates = []
        best_payload: dict[str, Any] | None = None

        for frame_combo in itertools.combinations(shared_frames, combo_size):
            shared_config = make_shared_frame_config(
                algorithm_params,
                str(source_npz),
                str(target_npz),
                list(frame_combo),
            )
            source_points, target_points, frame_names, frame_stats = collect_correspondences(
                source_batch,
                target_batch,
                shared_config,
            )
            candidate = ransac_rigid_candidates(
                source_points,
                target_points,
                frame_names,
                shared_config,
                source_frame=source_batch_id,
                target_frame=target_batch_id,
            )[0]
            metric_dict = candidate.metrics.to_dict()
            candidate = RegistrationCandidate(
                candidate_id=candidate.candidate_id,
                transform=Transform(
                    source=source_batch_id,
                    target=target_batch_id,
                    matrix=np.asarray(candidate.transform.matrix, dtype=float),
                ),
                score=float(candidate.score),
                metrics=candidate.metrics,
                metadata={
                    "frames": list(frame_combo),
                    "frame_stats": frame_stats,
                    **dict(candidate.metadata),
                },
            )
            row = {
                "frames": list(frame_combo),
                "score": float(candidate.score),
                "correspondence_count": int(len(source_points)),
                "coarse_threshold": float(metric_dict.get("coarse_threshold", 0.0)),
                "inlier_ratio": float(metric_dict.get("inlier_ratio", 0.0)),
                "inlier_count": int(metric_dict.get("inlier_count", 0)),
                "median_error": float(metric_dict.get("median_error", float("inf"))),
                "p90_error": float(metric_dict.get("p90_error", float("inf"))),
                "rmse": float(metric_dict.get("rmse", float("inf"))),
                "per_frame_metrics": metric_dict.get("per_frame_metrics", []),
            }
            rows.append(row)
            candidates.append(candidate)

            payload = {
                "row": row,
                "candidate": candidate,
                "source_points": source_points,
                "target_points": target_points,
                "frame_names": frame_names,
                "shared_frames": shared_frames,
            }
            if best_payload is None or row["score"] > best_payload["row"]["score"]:
                best_payload = payload

        if best_payload is None:
            raise RuntimeError("No valid frame-combination candidate was produced.")

        rows = sorted(rows, key=lambda item: item["score"], reverse=True)
        selected_frames = tuple(str(item) for item in best_payload["row"]["frames"])
        ranking = tuple(
            FrameSelectionCandidate(
                frames=tuple(str(item) for item in row["frames"]),
                score=float(row["score"]),
                correspondence_count=int(row["correspondence_count"]),
                metrics=EvaluationMetrics.from_mapping(row),
                raw_row=row,
            )
            for row in rows
        )
        frame_selection = FrameSelection(
            shared_frames=tuple(shared_frames),
            selected_frames=selected_frames,
            ranking=ranking,
        )
        sorted_candidates = tuple(sorted(candidates, key=lambda item: item.score, reverse=True))
        selected = best_payload["candidate"]

        return FrameSelectionPayload(
            registration=RegistrationResult(
                selected=selected,
                candidates=sorted_candidates,
                frame_selection=frame_selection,
                source_points=best_payload["source_points"],
                target_points=best_payload["target_points"],
                frame_names=best_payload["frame_names"],
                status="accepted",
            ),
            rows=rows,
        )


def evaluate_selected_shared_frames(
    registration: RegistrationResult,
    transform: Transform,
    evaluation_threshold: float,
) -> EvaluationMetrics:
    """用选中的共享帧 correspondence 评估某个 source->target 变换。"""

    metrics = evaluate_shared_correspondences(
        registration.source_points,
        registration.target_points,
        registration.frame_names,
        transform.matrix,
        float(evaluation_threshold),
    )
    return EvaluationMetrics.from_mapping(metrics)

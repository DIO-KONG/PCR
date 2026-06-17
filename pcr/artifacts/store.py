from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pcr.domain import PipelineResult, StepResult, Transform, WorldUpdateResult
from pcr.io.pointcloud_io import make_registration_overlay, save_point_cloud


def json_safe(value: Any) -> Any:
    """把 numpy、Path、dataclass、Transform 转为 JSON/YAML 可写对象。"""

    if isinstance(value, Transform):
        return value.to_json()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value


class ArtifactStore:
    """统一管理一次 run 的目录结构和文件写入。

    Pipeline 不再直接 `json.dumps`、`np.savetxt` 或保存 PLY；所有 artifact
    规则集中在这里，方便后续调整目录结构而不触碰算法。
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.transforms_dir = self.run_dir / "transforms"

    def initialize(self, *, config: dict[str, Any], overwrite: bool = False) -> None:
        if self.run_dir.exists() and not overwrite:
            raise FileExistsError(f"Run directory exists: {self.run_dir}. Use --overwrite.")
        if self.run_dir.exists() and overwrite:
            import shutil

            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.transforms_dir.mkdir(parents=True, exist_ok=True)
        self.write_yaml(self.run_dir / "run_config.yaml", config)

    def write_json(self, path: str | Path, payload: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")

    def write_yaml(self, path: str | Path, payload: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(json_safe(payload), allow_unicode=True, sort_keys=False), encoding="utf-8")

    def write_matrix(self, path: str | Path, transform_or_matrix: Transform | np.ndarray) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = transform_or_matrix.matrix if isinstance(transform_or_matrix, Transform) else transform_or_matrix
        np.savetxt(path, np.asarray(matrix, dtype=float), fmt="%.10f")

    def write_algorithm_record(self) -> None:
        record = (
            "# dynamic_top3_shared_frame_bounded_icp\n\n"
            "- 自动发现相邻 DA3 batch 共享帧，并枚举 4选3 粗配准。\n"
            "- 粗配准使用共享像素 3D 对应点、RANSAC、Kabsch/SVD，只估计刚体。\n"
            "- source 投到 global 后，以累计 world 为 target 做 bounded point-to-plane ICP。\n"
            "- ICP 未超过平移/旋转边界则采用；共享帧一致性只作为诊断。\n"
            "- 当前融合策略是 append + voxel downsample；复杂重影处理应作为独立 FusionPolicy。\n"
        )
        (self.run_dir / "algorithm_record.md").write_text(record, encoding="utf-8")

    def write_frame_combo_ranking(self, path: Path, rows: list[dict[str, Any]]) -> None:
        lines = [
            "| rank | frames | score | corr | inlier_ratio | median_m | p90_m | rmse_m |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for rank, row in enumerate(rows, 1):
            lines.append(
                "| {rank} | {frames} | {score:.6f} | {corr} | {ratio:.6f} | {median:.6f} | {p90:.6f} | {rmse:.6f} |".format(
                    rank=rank,
                    frames=", ".join(row["frames"]),
                    score=row["score"],
                    corr=row["correspondence_count"],
                    ratio=row["inlier_ratio"],
                    median=row["median_error"],
                    p90=row["p90_error"],
                    rmse=row["rmse"],
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_initial_transform(self, transform: Transform) -> None:
        self.write_matrix(self.transforms_dir / f"{transform.source}_to_{transform.target}.txt", transform)

    def write_step_with_world(
        self,
        update: WorldUpdateResult,
        *,
        world_before_step,
        frame_combo_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """写单步 artifact。

        overlay 需要“本步执行前的 world”作为灰色 target，因此由 sequence pipeline
        显式传入，避免 ArtifactStore 自己猜测状态。
        """

        step = update.step_result
        task = step.task
        step_dir = self.run_dir / f"step_{task.step_index:02d}_{task.source.batch_id}_to_{task.target.batch_id}"
        step_dir.mkdir(parents=True, exist_ok=True)

        save_point_cloud(step_dir / "coarse_overlay.ply", make_registration_overlay(world_before_step, step.coarse_registered_source))
        save_point_cloud(step_dir / "bounded_icp_overlay.ply", make_registration_overlay(world_before_step, step.refined_registered_source))
        save_point_cloud(step_dir / "final_registered_source.ply", step.final_registered_source)
        save_point_cloud(step_dir / "fused_world.ply", update.fused_world)
        self.write_matrix(step_dir / "coarse_matrix.txt", step.refinement.initial_transform)
        self.write_matrix(step_dir / "bounded_icp_matrix.txt", step.refinement.refined_transform)
        self.write_matrix(step_dir / "bounded_icp_delta_from_coarse.txt", step.refinement.delta_from_initial)
        self.write_matrix(step_dir / "final_matrix.txt", step.final_transform)
        self.write_matrix(self.transforms_dir / f"{task.source.batch_id}_to_global.txt", step.final_transform)
        self.write_json(
            step_dir / "selected_frames.json",
            {
                "shared_frames_discovered": step.coarse.frame_selection.shared_frames,
                "selected_frames": step.coarse.frame_selection.selected_frames,
                "selection": step.coarse.frame_selection.ranking[0].raw_row,
            },
        )
        self.write_frame_combo_ranking(step_dir / "frame_combo_ranking.md", frame_combo_rows)
        metrics = self.step_metrics(update, frame_combo_rows)
        self.write_json(step_dir / "metrics.json", metrics)
        return metrics

    def step_metrics(self, update: WorldUpdateResult, frame_combo_rows: list[dict[str, Any]]) -> dict[str, Any]:
        step = update.step_result
        task = step.task
        return {
            "step_index": task.step_index,
            "step": {
                "id": task.step_id,
                "source_id": task.source.batch_id,
                "target_id": task.target.batch_id,
                "source_npz": task.source.npz_path,
                "target_npz": task.target.npz_path,
                "source_cloud": task.source.preprocessed_cloud_path,
                "target_cloud": task.target.preprocessed_cloud_path,
            },
            "selected_frames": step.coarse.frame_selection.selected_frames,
            "shared_frames_discovered": step.coarse.frame_selection.shared_frames,
            "frame_combo_ranking": frame_combo_rows,
            "pairwise_matrix_source_to_target": step.coarse.selected.transform,
            "coarse_matrix_source_to_global": step.refinement.initial_transform,
            "bounded_icp_matrix_source_to_global": step.refinement.refined_transform,
            "bounded_icp_delta_from_coarse": step.refinement.delta_from_initial,
            "final_matrix_source_to_global": step.final_transform,
            "final_source": step.final_source,
            "coarse_metrics": step.coarse_shared_metrics.to_dict(),
            "icp_metrics": step.refinement.metrics.to_dict(),
            "source_cloud_points": step.source_cloud_points,
            "fused_world_points": update.fused_world_points,
            "world_stats": update.world_stats,
        }

    def write_summary(self, result: PipelineResult, summary_rows: list[dict[str, Any]], transforms: dict[str, Transform]) -> None:
        lines = [
            "# Shared-Frame Dynamic Top3 Walk-Forward Summary",
            "",
            "| step | pair | selected frames | coarse median m | icp status | icp fitness | icp rmse m | icp dt m | icp drot deg | final | fused points |",
            "|---:|---|---|---:|---|---:|---:|---:|---:|---|---:|",
        ]
        for row in summary_rows:
            lines.append(
                "| {step} | {source} -> {target} | {frames} | {coarse_median:.4f} | {icp_status} | {icp_fit:.4f} | {icp_rmse:.4f} | {dt:.4f} | {dr:.3f} | {used} | {points} |".format(
                    step=row["step_index"],
                    source=row["source_id"],
                    target=row["target_id"],
                    frames=", ".join(row["selected_frames"]),
                    coarse_median=row["coarse_metrics"]["median_error"],
                    icp_status=row["icp_metrics"]["status"],
                    icp_fit=row["icp_metrics"]["fitness"],
                    icp_rmse=row["icp_metrics"]["inlier_rmse"],
                    dt=row["icp_metrics"]["translation_delta"],
                    dr=row["icp_metrics"]["rotation_delta_deg"],
                    used=row["final_source"],
                    points=row["fused_world_points"],
                )
            )
        (self.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.write_json(
            self.run_dir / "summary.json",
            {
                "run_name": result.run_name,
                "run_dir": result.run_dir,
                "summary_rows": summary_rows,
                "final_world_points": result.final_world_points,
                "transforms": transforms,
            },
        )

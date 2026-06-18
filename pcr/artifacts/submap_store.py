from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pcr.artifacts.store import json_safe
from pcr.domain import SubmapPipelineResult, SubmapStepResult, Transform
from pcr.io.pointcloud_io import make_registration_overlay, save_point_cloud
from pcr.state.submap import SubmapManager


class SubmapArtifactStore:
    """online submap run 的 artifact 写入器。"""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.step_rows: list[dict[str, Any]] = []

    def initialize(self, *, config: dict[str, Any], overwrite: bool) -> None:
        if self.run_dir.exists() and not overwrite:
            raise FileExistsError(f"Run directory exists: {self.run_dir}. Use --overwrite.")
        if self.run_dir.exists() and overwrite:
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
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
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, np.asarray(matrix, dtype=float), fmt="%.10f")

    def write_step(self, step: SubmapStepResult) -> None:
        task = step.step_result.task
        step_dir = self.run_dir / "steps" / f"step_{task.step_index:03d}_{task.source.batch_id}_to_{task.target.batch_id}"
        step_dir.mkdir(parents=True, exist_ok=True)

        save_point_cloud(
            step_dir / "coarse_overlay.ply",
            make_registration_overlay(step.world_before_step, step.step_result.coarse_registered_source),
        )
        save_point_cloud(
            step_dir / "bounded_icp_overlay.ply",
            make_registration_overlay(step.world_before_step, step.step_result.refined_registered_source),
        )
        save_point_cloud(step_dir / "final_registered_source.ply", step.step_result.final_registered_source)
        self.write_matrix(step_dir / "coarse_matrix.txt", step.step_result.refinement.initial_transform)
        self.write_matrix(step_dir / "bounded_icp_matrix.txt", step.step_result.refinement.refined_transform)
        self.write_matrix(step_dir / "final_matrix.txt", step.step_result.final_transform)

        if step.fusion is not None:
            debug_dir = step_dir / "fusion_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            if not step.fusion.debug_clouds.accepted.is_empty():
                save_point_cloud(debug_dir / "accepted_points.ply", step.fusion.debug_clouds.accepted)
            if not step.fusion.debug_clouds.duplicate.is_empty():
                save_point_cloud(debug_dir / "duplicate_points.ply", step.fusion.debug_clouds.duplicate)
            if not step.fusion.debug_clouds.conflict.is_empty():
                save_point_cloud(debug_dir / "conflict_points.ply", step.fusion.debug_clouds.conflict)

        self.write_json(
            step_dir / "quality_report.json",
            {
                "active_submap_id": step.active_submap_id,
                "quality": step.quality,
                "coarse_metrics": step.step_result.coarse_shared_metrics.to_dict(),
                "icp_metrics": step.step_result.refinement.metrics.to_dict(),
            },
        )
        self.write_json(
            step_dir / "fusion_report.json",
            None if step.fusion is None else step.fusion.report,
        )
        self.write_json(
            step_dir / "frame_combo_ranking.json",
            step.frame_combo_rows,
        )
        self.step_rows.append(self.step_row(step))

    def step_row(self, step: SubmapStepResult) -> dict[str, Any]:
        """提取 summary 所需的轻量字段，避免长期保留完整点云。"""

        return {
            "step_index": step.step_result.task.step_index,
            "active_submap_id": step.active_submap_id,
            "source_id": step.step_result.task.source.batch_id,
            "target_id": step.step_result.task.target.batch_id,
            "selected_frames": step.step_result.coarse.frame_selection.selected_frames,
            "pose_status": step.quality.pose_status.value,
            "fusion_status": step.quality.fusion_status.value,
            "reasons": step.quality.reasons,
            "fusion": None if step.fusion is None else step.fusion.report,
        }

    def write_submaps(self, manager: SubmapManager) -> None:
        submap_root = self.run_dir / "submaps"
        for submap in manager.submaps:
            submap_dir = submap_root / submap.submap_id
            submap_dir.mkdir(parents=True, exist_ok=True)
            local_cloud = submap.local_cloud
            if not local_cloud.is_empty():
                save_point_cloud(submap_dir / "local_world.ply", local_cloud)
            transforms_dir = submap_dir / "transforms"
            for batch_id, transform in submap.transforms_to_submap.items():
                self.write_matrix(transforms_dir / f"{batch_id}_to_{submap.submap_id}.txt", transform)
            self.write_json(
                submap_dir / "metadata.json",
                {
                    "submap_id": submap.submap_id,
                    "anchor_batch_id": submap.anchor_batch_id,
                    "batch_order": submap.batch_order,
                    "step_count": submap.step_count,
                    "voxel_count": len(submap.fusion.voxel_map.voxels),
                    "local_point_count": int(len(local_cloud.points)),
                },
            )

    def write_summary(self, result: SubmapPipelineResult, manager: SubmapManager) -> None:
        self.write_submaps(manager)
        chain_payload = {
            "transforms_to_global": {} if manager.chain is None else manager.chain.transforms_to_global,
            "edges": [] if manager.chain is None else manager.chain.edges,
        }
        self.write_json(self.run_dir / "submap_chain.json", chain_payload)
        rows = self.step_rows or [self.step_row(step) for step in result.steps]
        self.write_json(
            self.run_dir / "summary.json",
            {
                "run_name": result.run_name,
                "run_dir": result.run_dir,
                "final_submap_id": result.final_submap_id,
                "global_preview_points": result.global_preview_points,
                "steps": rows,
            },
        )
        lines = [
            "# Online Submap Walk-Forward Summary",
            "",
            "| step | submap | pair | pose | fusion | new frames | conflict | reasons |",
            "|---:|---|---|---|---|---|---:|---|",
        ]
        for row in rows:
            fusion = row["fusion"]
            new_frames = "" if fusion is None else ", ".join(fusion.new_frame_names)
            conflict = 0.0 if fusion is None else fusion.conflict_ratio
            lines.append(
                "| {step} | {submap} | {source} -> {target} | {pose} | {fusion_status} | {frames} | {conflict:.3f} | {reasons} |".format(
                    step=row["step_index"],
                    submap=row["active_submap_id"],
                    source=row["source_id"],
                    target=row["target_id"],
                    pose=row["pose_status"],
                    fusion_status=row["fusion_status"],
                    frames=new_frames,
                    conflict=conflict,
                    reasons=", ".join(row["reasons"]) or "ok",
                )
            )
        (self.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

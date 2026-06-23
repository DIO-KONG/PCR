from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import open3d as o3d

from pcr.domain import QualityReport, QualityStatus, SequenceTask, SubmapPipelineResult, SubmapStepResult
from pcr.evaluation.quality import apply_fusion_quality, evaluate_pose_quality
from pcr.io.pointcloud_io import load_point_cloud, save_point_cloud
from pcr.pipeline.registration_step import RegistrationStepPipeline
from pcr.preprocessing.frame_cloud import build_new_frame_clouds
from pcr.preprocessing.pipeline import PreprocessService
from pcr.state.submap import SubmapManager
from pcr.state.world import WorldState


@dataclass(frozen=True)
class SubmapSequenceOutput:
    """online submap sequence 的完整输出。"""

    result: SubmapPipelineResult
    manager: SubmapManager


class SubmapSequencePipeline:
    """Online submap walk-forward pipeline。

    它沿用现有 shared-frame coarse + bounded ICP，但 ICP target 只使用 active
    submap。融合只使用当前 window 的新增帧，并通过 Voxel Hash 控制重影。
    """

    def __init__(
        self,
        *,
        preprocess_service: PreprocessService | None = None,
        step_pipeline: RegistrationStepPipeline | None = None,
    ) -> None:
        self.preprocess_service = preprocess_service or PreprocessService()
        self.step_pipeline = step_pipeline or RegistrationStepPipeline(preprocess_service=self.preprocess_service)

    def run(
        self,
        *,
        task: SequenceTask,
        config: dict[str, Any],
        run_name: str,
        run_dir: Path,
        step_callback: Callable[[SubmapStepResult], None] | None = None,
        keep_step_results: bool = False,
    ) -> SubmapSequenceOutput:
        submap_config = config.get("submap", {})
        fusion_config = config.get("fusion", {})
        quality_config = config.get("quality_gate", {})
        max_provisional_chain = int(submap_config.get("max_provisional_chain", 3))

        self.preprocess_service.ensure_floor_removed(task.baseline)
        baseline_cloud = load_point_cloud(task.baseline.preprocessed_cloud_path)
        manager = SubmapManager(
            submap_size=int(submap_config.get("submap_size", 10)),
            submap_overlap=int(submap_config.get("submap_overlap", 3)),
            fusion_config=fusion_config,
            min_submap_steps=int(submap_config.get("min_submap_steps", 8)),
            max_submap_steps=int(submap_config.get("max_submap_steps", 20)),
            min_fused_batches_for_rotation=int(submap_config.get("min_fused_batches_for_rotation", 5)),
            recent_fusion_window=int(submap_config.get("recent_fusion_window", 6)),
            min_recent_fused_for_rotation=int(submap_config.get("min_recent_fused_for_rotation", 2)),
            max_steps_since_fused_for_rotation=int(submap_config.get("max_steps_since_fused_for_rotation", 6)),
            max_submap_edge_translation=float(submap_config.get("max_submap_edge_translation", 5.0)),
            max_submap_edge_rotation_deg=float(submap_config.get("max_submap_edge_rotation_deg", 180.0)),
        )
        active = manager.initialize(baseline_id=task.baseline.batch_id, baseline_cloud=baseline_cloud)
        batch_cloud_paths: dict[str, str] = {task.baseline.batch_id: str(task.baseline.preprocessed_cloud_path)}
        step_outputs: list[SubmapStepResult] = []

        for step_task in task.steps:
            active = manager.active
            print(f"[submap {active.submap_id}] step {step_task.step_index}: {step_task.source.batch_id} -> {step_task.target.batch_id}", flush=True)
            self.preprocess_service.ensure_floor_removed(step_task.source)
            self.preprocess_service.ensure_floor_removed(step_task.target)
            batch_cloud_paths[step_task.source.batch_id] = str(step_task.source.preprocessed_cloud_path)
            batch_cloud_paths[step_task.target.batch_id] = str(step_task.target.preprocessed_cloud_path)
            target_provisional_depth = active.provisional_depth(step_task.target.batch_id)

            world_view = WorldState(
                world_cloud=active.local_cloud,
                transforms_to_global=dict(active.transforms_to_submap),
            )
            world_before = o3d.geometry.PointCloud(active.local_cloud)
            step_output = self.step_pipeline.run(step_task, world_view)
            step_result = step_output.step_result

            pose_quality = evaluate_pose_quality(step_result, quality_config)
            pose_quality = self.apply_provisional_chain_gate(
                pose_quality,
                target_provisional_depth=target_provisional_depth,
                max_provisional_chain=max_provisional_chain,
            )
            fusion_result = None
            final_quality = pose_quality

            if pose_quality.pose_status != QualityStatus.REJECTED:
                active.remember_registered_batch(step_task.source.batch_id, step_result.final_transform)
                sampling = step_task.params.get("sampling", {})
                frame_cloud, new_frame_names = build_new_frame_clouds(
                    step_task.source,
                    step_task.target,
                    conf_percentile=float(fusion_config.get("conf_percentile", sampling.get("conf_percentile", 20.0))),
                    stride=int(fusion_config.get("frame_point_stride", 1)),
                    max_points_per_frame=fusion_config.get("max_points_per_frame"),
                    random_seed=int(fusion_config.get("random_seed", sampling.get("random_seed", 7))),
                    floor_distance_threshold=float(fusion_config.get("floor_distance_threshold", 0.03)),
                    remove_floor=self.preprocess_service.remove_floor,
                )
                incoming = step_result.final_transform.apply_cloud(frame_cloud)
                preview_fusion = active.fusion.fuse(
                    incoming_cloud=incoming,
                    step_index=step_task.step_index,
                    new_frame_names=new_frame_names,
                    commit=False,
                )
                final_quality = apply_fusion_quality(pose_quality, preview_fusion.report, quality_config)
                # 方案 B 是点级过滤：accepted 写入，duplicate 更新，conflict 丢弃。
                # 但 fusion gate 需要先看“准备写入”的点是否像尺度阴影；命中时
                # 保留诊断 artifact，不提交到 voxel map，也不作为 overlap seed。
                if final_quality.fusion_status != QualityStatus.REJECTED:
                    fusion_result = active.fusion.fuse(
                        incoming_cloud=incoming,
                        step_index=step_task.step_index,
                        new_frame_names=new_frame_names,
                        commit=True,
                    )
                else:
                    fusion_result = preview_fusion

                # 只要本步确实改动了 voxel map，就记录为 fused，供 submap 切换时
                # 使用过滤后的 local map 作为 overlap seed。
                if final_quality.fusion_status != QualityStatus.REJECTED and fusion_result.report.accepted_points + fusion_result.report.duplicate_points > 0:
                    active.remember_fused_batch(step_task.source.batch_id)
            else:
                active.remember_provisional_batch(
                    step_task.source.batch_id,
                    step_result.final_transform,
                    depth=target_provisional_depth + 1,
                )

            active.step_count += 1
            active.record_step_outcome(
                committed_fusion=(
                    fusion_result is not None
                    and final_quality.fusion_status != QualityStatus.REJECTED
                    and bool(fusion_result.report.params.get("committed", True))
                    and fusion_result.report.accepted_points + fusion_result.report.duplicate_points > 0
                ),
                pose_rejected=final_quality.pose_status == QualityStatus.REJECTED,
                scale_quarantined="fusion_scale_shadow_quarantined" in final_quality.reasons,
            )
            submap_step = SubmapStepResult(
                step_result=step_result,
                quality=final_quality,
                fusion=fusion_result,
                active_submap_id=active.submap_id,
                world_before_step=world_before,
                frame_combo_rows=step_output.frame_combo_rows,
            )
            if step_callback is not None:
                step_callback(submap_step)
            if keep_step_results:
                step_outputs.append(submap_step)
            print(
                "[submap {sid}] pose={pose} fusion={fusion} reasons={reasons} points={points}".format(
                    sid=active.submap_id,
                    pose=final_quality.pose_status.value,
                    fusion=final_quality.fusion_status.value,
                    reasons=",".join(final_quality.reasons) or "ok",
                    points=len(active.local_cloud.points),
                ),
                flush=True,
            )
            before_submap = active.submap_id
            after = manager.rotate_if_needed(batch_cloud_paths)
            if after.submap_id == before_submap and active.rotation_hold_reason:
                print(f"[submap {active.submap_id}] rotation held: {active.rotation_hold_reason}", flush=True)

        preview = self.make_global_preview(manager)
        preview_path = run_dir / "global_preview.ply"
        if not preview.is_empty():
            save_point_cloud(preview_path, preview)
        result = SubmapPipelineResult(
            run_name=run_name,
            run_dir=run_dir,
            steps=tuple(step_outputs),
            submap_edges=tuple(manager.chain.edges if manager.chain is not None else ()),
            final_submap_id=manager.active.submap_id,
            global_preview_points=int(len(preview.points)),
        )
        return SubmapSequenceOutput(result=result, manager=manager)

    def make_global_preview(self, manager: SubmapManager) -> o3d.geometry.PointCloud:
        """把所有 submap 的 local cloud 按位姿链临时组合成全局预览。"""

        if manager.chain is None:
            return o3d.geometry.PointCloud()
        preview = o3d.geometry.PointCloud()
        for submap in manager.submaps:
            transform = manager.chain.transforms_to_global[submap.submap_id]
            preview += transform.apply_cloud(submap.local_cloud)
        voxel_size = float(manager.fusion_config.get("global_preview_voxel_size", manager.fusion_config.get("voxel_size", 0.06)))
        return preview.voxel_down_sample(voxel_size) if voxel_size > 0 and not preview.is_empty() else preview

    def apply_provisional_chain_gate(
        self,
        report: QualityReport,
        *,
        target_provisional_depth: int,
        max_provisional_chain: int,
    ) -> QualityReport:
        """限制 rejected 位姿链继续污染 ICP/fusion。

        shared-frame coarse 仍然会计算；这里的 gate 只阻止过深 provisional
        transform 进入地图和 overlap seed。
        """

        metrics = dict(report.metrics)
        metrics["target_provisional_depth"] = int(target_provisional_depth)
        metrics["max_provisional_chain"] = int(max_provisional_chain)
        if target_provisional_depth <= max_provisional_chain:
            return QualityReport(
                pose_status=report.pose_status,
                fusion_status=report.fusion_status,
                reasons=report.reasons,
                metrics=metrics,
            )
        reasons = tuple([*report.reasons, "target_provisional_chain_too_deep"])
        return QualityReport(
            pose_status=QualityStatus.REJECTED,
            fusion_status=QualityStatus.REJECTED,
            reasons=reasons,
            metrics=metrics,
        )

from __future__ import annotations

from dataclasses import dataclass

import open3d as o3d

from pcr.domain import PipelineResult, SequenceTask, WorldUpdateResult
from pcr.io.pointcloud_io import load_point_cloud, save_point_cloud
from pcr.pipeline.registration_step import RegistrationStepPipeline
from pcr.preprocessing.pipeline import PreprocessService
from pcr.state.fusion import AppendVoxelFusion
from pcr.state.world import WorldState


@dataclass(frozen=True)
class SequenceStepOutput:
    """sequence 中一步执行后交给 app/artifact 的数据。"""

    update: WorldUpdateResult
    frame_combo_rows: list[dict]
    world_before_step: o3d.geometry.PointCloud


@dataclass(frozen=True)
class SequencePipelineOutput:
    """完整 sequence 输出。"""

    result: PipelineResult
    steps: tuple[SequenceStepOutput, ...]
    world: WorldState


class SequencePipeline:
    """完整 walk-forward pipeline。

    它负责按照配置顺序调用单步 pipeline，并用 WorldState 更新累计地图。
    它不写 artifact；app 层可以在每步后把结构化结果交给 ArtifactStore。
    """

    def __init__(
        self,
        *,
        preprocess_service: PreprocessService | None = None,
        step_pipeline: RegistrationStepPipeline | None = None,
    ) -> None:
        self.preprocess_service = preprocess_service or PreprocessService()
        self.step_pipeline = step_pipeline or RegistrationStepPipeline(preprocess_service=self.preprocess_service)

    def initialize_world(self, task: SequenceTask) -> WorldState:
        self.preprocess_service.ensure_floor_removed(task.baseline)
        baseline_cloud = load_point_cloud(task.baseline.preprocessed_cloud_path)
        return WorldState.initialize(baseline_id=task.baseline.batch_id, baseline_cloud=baseline_cloud)

    def run(self, *, task: SequenceTask, run_name: str, run_dir) -> SequencePipelineOutput:
        world = self.initialize_world(task)
        fusion = AppendVoxelFusion(voxel_size=float(task.params["fusion"]["voxel_size"]))
        outputs: list[SequenceStepOutput] = []

        for step_task in task.steps:
            print(f"[step {step_task.step_index}] {step_task.source.batch_id} -> {step_task.target.batch_id}", flush=True)
            world_before = o3d.geometry.PointCloud(world.world_cloud)
            step_output = self.step_pipeline.run(step_task, world)
            update = world.apply_step(step_output.step_result, fusion)
            outputs.append(
                SequenceStepOutput(
                    update=update,
                    frame_combo_rows=step_output.frame_combo_rows,
                    world_before_step=world_before,
                )
            )
            metrics = step_output.step_result
            print(
                "[step {idx}] frames={frames} coarse_median={median:.4f} icp={status} dt={dt:.4f} drot={dr:.3f} world_points={pts}".format(
                    idx=step_task.step_index,
                    frames=",".join(metrics.coarse.frame_selection.selected_frames),
                    median=metrics.coarse_shared_metrics.median_error,
                    status=metrics.refinement.metrics.status,
                    dt=metrics.refinement.metrics.translation_delta,
                    dr=metrics.refinement.metrics.rotation_delta_deg,
                    pts=update.fused_world_points,
                ),
                flush=True,
            )

        final_world_path = run_dir / "final_world.ply"
        save_point_cloud(final_world_path, world.world_cloud)
        result = PipelineResult(
            run_name=run_name,
            run_dir=run_dir,
            steps=tuple(item.update for item in outputs),
            final_world_points=int(len(world.world_cloud.points)),
        )
        return SequencePipelineOutput(result=result, steps=tuple(outputs), world=world)

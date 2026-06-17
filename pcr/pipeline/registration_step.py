from __future__ import annotations

from dataclasses import dataclass

from pcr.algorithms.refinement.icp import BoundedPointToPlaneIcp
from pcr.algorithms.shared_frame.frame_selection import (
    DynamicTopKSharedFrameSelector,
    evaluate_selected_shared_frames,
)
from pcr.domain import RegistrationTask, StepResult
from pcr.io.pointcloud_io import load_point_cloud
from pcr.preprocessing.pipeline import PreprocessService
from pcr.state.world import WorldState


@dataclass(frozen=True)
class StepPipelineOutput:
    """单步 pipeline 输出。

    `frame_combo_rows` 只用于 artifact 报告，不参与后续算法决策。
    """

    step_result: StepResult
    frame_combo_rows: list[dict]


class RegistrationStepPipeline:
    """执行一次 source batch 到当前 global world 的配准。

    本类只编排单步流程：
    预处理检查 -> 动态共享帧粗配准 -> source->global 初值组合 ->
    bounded ICP -> 构造 StepResult。

    它不保存文件、不更新 world，也不生成 Markdown。
    """

    def __init__(
        self,
        *,
        preprocess_service: PreprocessService | None = None,
        frame_selector: DynamicTopKSharedFrameSelector | None = None,
        icp_refiner: BoundedPointToPlaneIcp | None = None,
    ) -> None:
        self.preprocess_service = preprocess_service or PreprocessService()
        self.frame_selector = frame_selector or DynamicTopKSharedFrameSelector()
        self.icp_refiner = icp_refiner or BoundedPointToPlaneIcp()

    def run(self, task: RegistrationTask, world: WorldState) -> StepPipelineOutput:
        self.preprocess_service.ensure_floor_removed(task.source)
        self.preprocess_service.ensure_floor_removed(task.target)
        source_cloud = load_point_cloud(task.source.preprocessed_cloud_path)

        selection_payload = self.frame_selector.select(
            source_batch_id=task.source.batch_id,
            target_batch_id=task.target.batch_id,
            source_npz=str(task.source.npz_path),
            target_npz=str(task.target.npz_path),
            algorithm_params=task.params,
        )
        coarse = selection_payload.registration

        # coarse.selected.transform 是 source -> target；target_to_global 是 target -> global。
        # 使用 Transform.then() 后得到 source -> global，并由 Transform 校验方向。
        target_to_global = world.transform_to_global(task.target.batch_id)
        coarse_global = coarse.selected.transform.then(target_to_global)
        coarse_registered = coarse_global.apply_cloud(source_cloud)

        refinement = self.icp_refiner.refine(
            source_cloud=source_cloud,
            target_world=world.world_cloud,
            initial_transform=coarse_global,
            icp_config=task.params["icp"],
        )
        final_transform = refinement.accepted_transform
        final_source = "bounded_icp" if refinement.accepted else "coarse"
        refined_registered = refinement.refined_transform.apply_cloud(source_cloud)
        final_registered = final_transform.apply_cloud(source_cloud)

        coarse_metrics = evaluate_selected_shared_frames(
            coarse,
            coarse.selected.transform,
            float(task.params["ransac"]["evaluation_threshold"]),
        )

        return StepPipelineOutput(
            step_result=StepResult(
                task=task,
                coarse=coarse,
                refinement=refinement,
                final_transform=final_transform,
                final_source=final_source,
                final_registered_source=final_registered,
                coarse_registered_source=coarse_registered,
                refined_registered_source=refined_registered,
                coarse_shared_metrics=coarse_metrics,
                source_cloud_points=int(len(source_cloud.points)),
            ),
            frame_combo_rows=selection_payload.rows,
        )


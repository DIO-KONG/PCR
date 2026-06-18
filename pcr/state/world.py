from __future__ import annotations

from dataclasses import dataclass, field

import open3d as o3d
import numpy as np

from pcr.domain import StepResult, Transform, WorldUpdateResult
from pcr.state.fusion import AppendVoxelFusion
from pcr.preprocessing.preprocess import describe_point_cloud


@dataclass
class WorldState:
    """walk-forward 过程中的全局状态。

    该对象是 world 点云和 batch->global transform 的唯一所有者。pipeline
    通过它查询 target 的 global 位姿，并在 step 成功后更新世界。
    """

    world_cloud: o3d.geometry.PointCloud
    transforms_to_global: dict[str, Transform]
    history: list[StepResult] = field(default_factory=list)

    @classmethod
    def initialize(cls, *, baseline_id: str, baseline_cloud: o3d.geometry.PointCloud) -> "WorldState":
        return cls(
            world_cloud=baseline_cloud,
            # global 坐标系以 baseline 初始 world 为基准；矩阵是单位阵，但
            # target frame 明确写成 global，后续组合出的矩阵语义就是
            # `batch_x -> global`。
            transforms_to_global={baseline_id: Transform(source=baseline_id, target="global", matrix=np.eye(4))},
        )

    def transform_to_global(self, batch_id: str) -> Transform:
        """读取某个 batch 到 global 的变换。"""

        if batch_id not in self.transforms_to_global:
            raise KeyError(f"No transform to global for batch: {batch_id}")
        return self.transforms_to_global[batch_id]

    def apply_step(self, step_result: StepResult, fusion: AppendVoxelFusion) -> WorldUpdateResult:
        """将单步结果融合进 world，并记录 source 的 global 位姿。"""

        fused_world = fusion.fuse(
            world_cloud=self.world_cloud,
            registered_source=step_result.final_registered_source,
        )
        self.world_cloud = fused_world
        self.transforms_to_global[step_result.task.source.batch_id] = step_result.final_transform
        self.history.append(step_result)
        return WorldUpdateResult(
            step_result=step_result,
            fused_world=fused_world,
            fused_world_points=int(len(fused_world.points)),
            world_stats=describe_point_cloud(fused_world),
        )

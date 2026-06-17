from __future__ import annotations

from dataclasses import dataclass

import open3d as o3d


@dataclass(frozen=True)
class AppendVoxelFusion:
    """最简单、可解释的融合策略。

    当前语义与旧 runner 保持一致：把变换后的 source 加到 world，再做一次体素
    降采样。复杂的重影/冲突过滤应作为新的 FusionPolicy 实现，而不是写进
    sequence pipeline。
    """

    voxel_size: float

    def fuse(
        self,
        *,
        world_cloud: o3d.geometry.PointCloud,
        registered_source: o3d.geometry.PointCloud,
    ) -> o3d.geometry.PointCloud:
        fused = world_cloud + registered_source
        return fused.voxel_down_sample(float(self.voxel_size))


from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from pcr.domain import SubmapEdge, Transform
from pcr.algorithms.shared_frame.geometry import rotation_angle_deg
from pcr.state.fusion import ConservativeVoxelHashFusion


@dataclass
class SubmapState:
    """一个局部 submap。

    submap 坐标系可以和某个 anchor batch 坐标系一致。内部维护 batch 到该
    submap 的 transform，以及用于 bounded ICP 的局部点云。
    """

    submap_id: str
    anchor_batch_id: str
    fusion: ConservativeVoxelHashFusion
    transforms_to_submap: dict[str, Transform]
    # registered_batch_order：已经通过 pose gate、拥有可追踪 transform 的 batch。
    # fused_batch_order：真正写入 voxel map、可作为下一个 submap overlap seed 的 batch。
    # provisional_batch_order：pose rejected 但为了相邻链路继续试算而保留的临时 transform。
    registered_batch_order: list[str] = field(default_factory=list)
    fused_batch_order: list[str] = field(default_factory=list)
    provisional_batch_order: list[str] = field(default_factory=list)
    batch_order: list[str] = field(default_factory=list)
    provisional_depths: dict[str, int] = field(default_factory=dict)
    fusion_step_history: list[bool] = field(default_factory=list)
    pose_rejected_history: list[bool] = field(default_factory=list)
    scale_quarantine_history: list[bool] = field(default_factory=list)
    rotation_hold_reason: str = ""
    step_count: int = 0

    @property
    def local_cloud(self) -> o3d.geometry.PointCloud:
        return self.fusion.voxel_map.to_cloud()

    def transform_to_submap(self, batch_id: str) -> Transform:
        if batch_id not in self.transforms_to_submap:
            raise KeyError(f"No transform to {self.submap_id} for batch: {batch_id}")
        return self.transforms_to_submap[batch_id]

    def remember_registered_batch(self, batch_id: str, transform: Transform) -> None:
        """记录 pose accepted 的 batch。

        这只表示 batch 有可追踪位姿，不代表它已经写入地图，也不代表可作为
        新 submap 的 overlap seed。
        """

        self.transforms_to_submap[batch_id] = transform
        self.provisional_depths.pop(batch_id, None)
        if batch_id not in self.registered_batch_order:
            self.registered_batch_order.append(batch_id)
        if batch_id in self.provisional_batch_order:
            self.provisional_batch_order.remove(batch_id)
        if batch_id not in self.batch_order:
            self.batch_order.append(batch_id)

    def remember_provisional_batch(self, batch_id: str, transform: Transform, *, depth: int) -> None:
        """记录 pose rejected 的临时位姿。

        这只用于让下一步相邻 batch 还能组合初值继续试算；它不会进入
        registered/fused，也不会作为 overlap seed。`depth` 表示这个临时位姿
        已经从最近可信 transform 传播了多少步，后续质量门控会限制它继续污染
        地图。
        """

        self.transforms_to_submap[batch_id] = transform
        self.provisional_depths[batch_id] = int(depth)
        if batch_id not in self.provisional_batch_order:
            self.provisional_batch_order.append(batch_id)
        if batch_id not in self.batch_order:
            self.batch_order.append(batch_id)

    def remember_fused_batch(self, batch_id: str) -> None:
        """记录 fusion accepted 的 batch，可用于 overlap seed。"""

        self.provisional_depths.pop(batch_id, None)
        if batch_id in self.provisional_batch_order:
            self.provisional_batch_order.remove(batch_id)
        if batch_id not in self.fused_batch_order:
            self.fused_batch_order.append(batch_id)
        if batch_id not in self.batch_order:
            self.batch_order.append(batch_id)

    def provisional_depth(self, batch_id: str) -> int:
        """返回 batch 的 provisional 传播深度；可信 batch 深度为 0。"""

        return int(self.provisional_depths.get(batch_id, 0))

    def record_step_outcome(self, *, committed_fusion: bool, pose_rejected: bool, scale_quarantined: bool) -> None:
        """记录当前 submap 内的真实增长情况，用于质量驱动切换。"""

        self.fusion_step_history.append(bool(committed_fusion))
        self.pose_rejected_history.append(bool(pose_rejected))
        self.scale_quarantine_history.append(bool(scale_quarantined))

    def steps_since_last_fusion(self) -> int:
        """距离最近一次实际写入 voxel map 的步数。"""

        for offset, item in enumerate(reversed(self.fusion_step_history), start=0):
            if item:
                return offset
        return len(self.fusion_step_history)


@dataclass
class SubmapChain:
    """维护 submap 到 global 的链式位姿。"""

    transforms_to_global: dict[str, Transform]
    edges: list[SubmapEdge] = field(default_factory=list)

    @classmethod
    def initialize(cls, first_submap_id: str) -> "SubmapChain":
        return cls(
            transforms_to_global={
                first_submap_id: Transform(source=first_submap_id, target="global", matrix=np.eye(4))
            }
        )

    def add_edge(self, edge: SubmapEdge) -> None:
        target_to_global = self.transforms_to_global[edge.target_submap_id]
        self.transforms_to_global[edge.source_submap_id] = edge.transform.then(target_to_global)
        self.edges.append(edge)


@dataclass
class SubmapManager:
    """Online submap 管理器。

    第一版使用固定 `submap_size + submap_overlap`。新 submap 以最近一个 overlap
    batch 作为局部坐标 anchor，并把最近 overlap 个 batch 作为 ICP seed。
    """

    submap_size: int
    submap_overlap: int
    fusion_config: dict
    min_submap_steps: int = 8
    max_submap_steps: int = 20
    min_fused_batches_for_rotation: int = 5
    recent_fusion_window: int = 6
    min_recent_fused_for_rotation: int = 2
    max_steps_since_fused_for_rotation: int = 6
    max_submap_edge_translation: float = 5.0
    max_submap_edge_rotation_deg: float = 180.0
    submaps: list[SubmapState] = field(default_factory=list)
    chain: SubmapChain | None = None

    def initialize(self, *, baseline_id: str, baseline_cloud: o3d.geometry.PointCloud) -> SubmapState:
        submap_id = "submap_000"
        fusion = ConservativeVoxelHashFusion.from_initial_cloud(baseline_cloud, config=self.fusion_config)
        baseline_transform = Transform(source=baseline_id, target=submap_id, matrix=np.eye(4))
        submap = SubmapState(
            submap_id=submap_id,
            anchor_batch_id=baseline_id,
            fusion=fusion,
            transforms_to_submap={baseline_id: baseline_transform},
            registered_batch_order=[baseline_id],
            fused_batch_order=[baseline_id],
            provisional_batch_order=[],
            batch_order=[baseline_id],
            provisional_depths={},
        )
        self.submaps = [submap]
        self.chain = SubmapChain.initialize(submap_id)
        return submap

    @property
    def active(self) -> SubmapState:
        if not self.submaps:
            raise RuntimeError("SubmapManager is not initialized.")
        return self.submaps[-1]

    def should_rotate(self) -> bool:
        active = self.active
        if active.step_count < int(self.submap_size):
            active.rotation_hold_reason = ""
            return False
        recent = active.fusion_step_history[-int(self.recent_fusion_window) :]
        fused_count = int(sum(active.fusion_step_history))
        recent_fused = int(sum(recent))
        steps_since_fused = active.steps_since_last_fusion()
        enough_growth = (
            active.step_count >= int(self.min_submap_steps)
            and fused_count >= int(self.min_fused_batches_for_rotation)
            and recent_fused >= int(self.min_recent_fused_for_rotation)
            and steps_since_fused <= int(self.max_steps_since_fused_for_rotation)
        )
        if enough_growth:
            active.rotation_hold_reason = ""
            return True
        if active.step_count >= int(self.max_submap_steps):
            active.rotation_hold_reason = (
                "hold_rotation_without_reliable_fusion:"
                f"steps={active.step_count},fused={fused_count},"
                f"recent_fused={recent_fused},steps_since_fused={steps_since_fused}"
            )
        return False

    def rotate_if_needed(self, batch_cloud_paths: dict[str, str]) -> SubmapState:
        if not self.should_rotate():
            return self.active
        return self.create_next_submap(batch_cloud_paths)

    def create_next_submap(self, batch_cloud_paths: dict[str, str]) -> SubmapState:
        previous = self.active
        seed_batches = previous.fused_batch_order[-int(self.submap_overlap) :]
        if not seed_batches:
            raise RuntimeError("Cannot create next submap without fusion-accepted overlap batches.")

        anchor_batch = seed_batches[-1]
        anchor_to_previous = previous.transform_to_submap(anchor_batch)
        edge_translation = float(np.linalg.norm(anchor_to_previous.matrix[:3, 3]))
        edge_rotation = rotation_angle_deg(anchor_to_previous.matrix[:3, :3])
        if (
            edge_translation > float(self.max_submap_edge_translation)
            or edge_rotation > float(self.max_submap_edge_rotation_deg)
        ):
            previous.rotation_hold_reason = (
                "hold_rotation_edge_sanity_failed:"
                f"anchor={anchor_batch},translation={edge_translation:.3f},rotation_deg={edge_rotation:.3f}"
            )
            return previous

        previous_to_anchor = anchor_to_previous.inverse()
        new_index = len(self.submaps)
        new_submap_id = f"submap_{new_index:03d}"

        transforms: dict[str, Transform] = {}
        registered_order: list[str] = []
        fused_order: list[str] = []
        provisional_order: list[str] = []
        provisional_depths: dict[str, int] = {}
        batch_order: list[str] = []

        def remember_order(target: list[str], batch_id: str) -> None:
            if batch_id not in target:
                target.append(batch_id)
            if batch_id not in batch_order:
                batch_order.append(batch_id)

        def carry_transform(batch_id: str) -> Transform:
            batch_to_previous = previous.transform_to_submap(batch_id)
            batch_to_new = batch_to_previous.then(previous_to_anchor)
            transform = Transform(
                source=batch_id,
                target=new_submap_id,
                matrix=batch_to_new.matrix,
            )
            transforms[batch_id] = transform
            return transform

        for batch_id in seed_batches:
            transform = carry_transform(batch_id)
            remember_order(registered_order, batch_id)
            remember_order(fused_order, batch_id)

        # 方案 B 的主地图已经是 accepted/duplicate 过滤后的 voxel map。
        # 新 submap 用过滤后的 local map 做 seed，而不是重新加载完整 batch PLY，
        # 否则之前丢弃的 conflict 点会在 submap 切换时被带回来。
        seed_cloud = previous_to_anchor.apply_cloud(previous.local_cloud)
        if seed_cloud.is_empty():
            raise RuntimeError(f"Failed to seed {new_submap_id}; overlap batches have no loadable clouds.")

        # 地图 seed 只来自 fused batch；但为了相邻链路不中断，需要把最近若干
        # 非融合 batch 的 transform 也带入新 submap。它们不会写入 voxel map，
        # 也不会成为下一次 overlap seed。
        carry_batches = previous.batch_order[-int(self.submap_overlap) :]
        for batch_id in carry_batches:
            if batch_id not in previous.transforms_to_submap:
                continue
            if batch_id not in transforms:
                carry_transform(batch_id)
            if batch_id in previous.registered_batch_order:
                remember_order(registered_order, batch_id)
            elif batch_id in previous.provisional_batch_order:
                remember_order(provisional_order, batch_id)
                provisional_depths[batch_id] = previous.provisional_depth(batch_id)
            elif batch_id not in batch_order:
                batch_order.append(batch_id)

        fusion = ConservativeVoxelHashFusion.from_initial_cloud(seed_cloud, config=self.fusion_config)
        submap = SubmapState(
            submap_id=new_submap_id,
            anchor_batch_id=anchor_batch,
            fusion=fusion,
            transforms_to_submap=transforms,
            registered_batch_order=registered_order,
            fused_batch_order=fused_order,
            provisional_batch_order=provisional_order,
            batch_order=batch_order,
            provisional_depths=provisional_depths,
        )
        self.submaps.append(submap)

        edge = SubmapEdge(
            source_submap_id=new_submap_id,
            target_submap_id=previous.submap_id,
            transform=Transform(
                source=new_submap_id,
                target=previous.submap_id,
                matrix=anchor_to_previous.matrix,
            ),
            status="overlap_seed",
            reason=f"anchor={anchor_batch}",
        )
        if self.chain is None:
            raise RuntimeError("Submap chain is not initialized.")
        self.chain.add_edge(edge)
        return submap

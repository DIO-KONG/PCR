from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from pcr.domain import SubmapEdge, Transform
from pcr.io.pointcloud_io import load_point_cloud
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
    registered_batch_order: list[str] = field(default_factory=list)
    fused_batch_order: list[str] = field(default_factory=list)
    batch_order: list[str] = field(default_factory=list)
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
        if batch_id not in self.registered_batch_order:
            self.registered_batch_order.append(batch_id)
        if batch_id not in self.batch_order:
            self.batch_order.append(batch_id)

    def remember_fused_batch(self, batch_id: str) -> None:
        """记录 fusion accepted 的 batch，可用于 overlap seed。"""

        if batch_id not in self.fused_batch_order:
            self.fused_batch_order.append(batch_id)
        if batch_id not in self.batch_order:
            self.batch_order.append(batch_id)


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
            batch_order=[baseline_id],
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
        return self.active.step_count >= int(self.submap_size)

    def rotate_if_needed(self, batch_cloud_paths: dict[str, str]) -> SubmapState:
        if not self.should_rotate():
            return self.active
        return self.create_next_submap(batch_cloud_paths)

    def create_next_submap(self, batch_cloud_paths: dict[str, str]) -> SubmapState:
        previous = self.active
        overlap_batches = previous.fused_batch_order[-int(self.submap_overlap) :]
        if not overlap_batches:
            raise RuntimeError("Cannot create next submap without fusion-accepted overlap batches.")

        anchor_batch = overlap_batches[-1]
        anchor_to_previous = previous.transform_to_submap(anchor_batch)
        previous_to_anchor = anchor_to_previous.inverse()
        new_index = len(self.submaps)
        new_submap_id = f"submap_{new_index:03d}"

        seed_cloud = o3d.geometry.PointCloud()
        transforms: dict[str, Transform] = {}
        for batch_id in overlap_batches:
            if batch_id not in batch_cloud_paths:
                continue
            batch_to_previous = previous.transform_to_submap(batch_id)
            batch_to_new = batch_to_previous.then(previous_to_anchor)
            transforms[batch_id] = Transform(
                source=batch_id,
                target=new_submap_id,
                matrix=batch_to_new.matrix,
            )
            cloud = load_point_cloud(batch_cloud_paths[batch_id])
            seed_cloud += transforms[batch_id].apply_cloud(cloud)

        if seed_cloud.is_empty():
            raise RuntimeError(f"Failed to seed {new_submap_id}; overlap batches have no loadable clouds.")

        fusion = ConservativeVoxelHashFusion.from_initial_cloud(seed_cloud, config=self.fusion_config)
        submap = SubmapState(
            submap_id=new_submap_id,
            anchor_batch_id=anchor_batch,
            fusion=fusion,
            transforms_to_submap=transforms,
            registered_batch_order=list(overlap_batches),
            fused_batch_order=list(overlap_batches),
            batch_order=list(overlap_batches),
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

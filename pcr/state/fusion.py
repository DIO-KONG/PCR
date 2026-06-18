from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d

from pcr.domain import FusionDebugClouds, FusionReport, FusionResult


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


@dataclass
class VoxelRecord:
    """Voxel Hash 中一个体素的稳定观测。"""

    centroid: np.ndarray
    color: np.ndarray
    normal: np.ndarray
    count: int = 1
    last_seen_step: int = 0

    def update(self, point: np.ndarray, color: np.ndarray, normal: np.ndarray, *, step_index: int) -> None:
        """用一致的重复观测更新体素均值。"""

        weight = float(self.count)
        self.centroid = (self.centroid * weight + point) / (weight + 1.0)
        self.color = (self.color * weight + color) / (weight + 1.0)
        normal_sum = self.normal * weight + normal
        norm = float(np.linalg.norm(normal_sum))
        self.normal = normal_sum / norm if norm > 1e-12 else self.normal
        self.count += 1
        self.last_seen_step = int(step_index)


@dataclass
class VoxelHashMap:
    """简单 Voxel Hash 地图。

    第一版以可解释和可调试为主：体素负责控制密度，nearest-neighbor 负责判断
    新点和现有地图之间是否重复或冲突。
    """

    voxel_size: float
    voxels: dict[tuple[int, int, int], VoxelRecord] = field(default_factory=dict)

    def key(self, point: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(point / float(self.voxel_size)).astype(int).tolist())

    def insert_or_update(
        self,
        point: np.ndarray,
        color: np.ndarray,
        normal: np.ndarray,
        *,
        step_index: int,
    ) -> None:
        voxel_key = self.key(point)
        if voxel_key in self.voxels:
            self.voxels[voxel_key].update(point, color, normal, step_index=step_index)
        else:
            self.voxels[voxel_key] = VoxelRecord(
                centroid=np.asarray(point, dtype=float),
                color=np.asarray(color, dtype=float),
                normal=np.asarray(normal, dtype=float),
                count=1,
                last_seen_step=int(step_index),
            )

    @classmethod
    def from_cloud(cls, cloud: o3d.geometry.PointCloud, *, voxel_size: float) -> "VoxelHashMap":
        result = cls(voxel_size=float(voxel_size))
        normalized = ensure_normals(cloud)
        points = np.asarray(normalized.points)
        colors = point_colors(normalized)
        normals = np.asarray(normalized.normals)
        for point, color, normal in zip(points, colors, normals, strict=False):
            result.insert_or_update(point, color, normal, step_index=0)
        return result

    def to_cloud(self) -> o3d.geometry.PointCloud:
        cloud = o3d.geometry.PointCloud()
        if not self.voxels:
            return cloud
        records = list(self.voxels.values())
        cloud.points = o3d.utility.Vector3dVector(np.vstack([item.centroid for item in records]))
        cloud.colors = o3d.utility.Vector3dVector(np.vstack([item.color for item in records]))
        cloud.normals = o3d.utility.Vector3dVector(np.vstack([item.normal for item in records]))
        return cloud


def point_colors(cloud: o3d.geometry.PointCloud) -> np.ndarray:
    """返回点颜色；无颜色时使用中性灰。"""

    points = np.asarray(cloud.points)
    if cloud.has_colors():
        return np.asarray(cloud.colors)
    return np.full((len(points), 3), 0.75, dtype=float)


def ensure_normals(cloud: o3d.geometry.PointCloud, *, radius: float = 0.18, max_nn: int = 30) -> o3d.geometry.PointCloud:
    """确保点云带法线，避免 fusion gate 缺少方向信息。"""

    result = o3d.geometry.PointCloud(cloud)
    if not result.has_normals() and not result.is_empty():
        result.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    return result


def make_debug_cloud(points: list[np.ndarray], colors: list[np.ndarray], paint: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    """把分类点转成调试点云；空分类返回空点云。"""

    cloud = o3d.geometry.PointCloud()
    if not points:
        return cloud
    cloud.points = o3d.utility.Vector3dVector(np.vstack(points))
    if colors:
        cloud.colors = o3d.utility.Vector3dVector(np.vstack(colors))
    else:
        cloud.paint_uniform_color(paint)
    return cloud


@dataclass
class ConservativeVoxelHashFusion:
    """保守 Voxel Hash 融合。

    目标是让地图密度稳定，并把 DA3 局部畸变导致的重影点挡在主地图之外。
    已有地图优先；新点只补充未知区域或更新一致重复观测。
    """

    voxel_map: VoxelHashMap
    duplicate_distance: float = 0.04
    conflict_distance: float = 0.15
    normal_angle_deg: float = 25.0
    normal_radius: float = 0.18
    normal_max_nn: int = 30

    @classmethod
    def from_initial_cloud(cls, cloud: o3d.geometry.PointCloud, *, config: dict) -> "ConservativeVoxelHashFusion":
        voxel_size = float(config.get("voxel_size", 0.06))
        return cls(
            voxel_map=VoxelHashMap.from_cloud(cloud, voxel_size=voxel_size),
            duplicate_distance=float(config.get("duplicate_distance", 0.04)),
            conflict_distance=float(config.get("conflict_distance", 0.15)),
            normal_angle_deg=float(config.get("normal_angle_deg", 25.0)),
            normal_radius=float(config.get("normal_radius", 0.18)),
            normal_max_nn=int(config.get("normal_max_nn", 30)),
        )

    def fuse(
        self,
        *,
        incoming_cloud: o3d.geometry.PointCloud,
        step_index: int,
        new_frame_names: tuple[str, ...],
    ) -> FusionResult:
        incoming = ensure_normals(incoming_cloud, radius=self.normal_radius, max_nn=self.normal_max_nn)
        map_cloud = self.voxel_map.to_cloud()
        map_cloud = ensure_normals(map_cloud, radius=self.normal_radius, max_nn=self.normal_max_nn)

        source_points = np.asarray(incoming.points)
        source_colors = point_colors(incoming)
        source_normals = np.asarray(incoming.normals) if incoming.has_normals() else np.zeros_like(source_points)
        map_points = np.asarray(map_cloud.points)
        map_normals = np.asarray(map_cloud.normals) if map_cloud.has_normals() else np.zeros_like(map_points)

        accepted_points: list[np.ndarray] = []
        accepted_colors: list[np.ndarray] = []
        duplicate_points: list[np.ndarray] = []
        duplicate_colors: list[np.ndarray] = []
        conflict_points: list[np.ndarray] = []
        conflict_colors: list[np.ndarray] = []

        tree = o3d.geometry.KDTreeFlann(map_cloud) if len(map_points) else None
        cos_threshold = float(np.cos(np.deg2rad(self.normal_angle_deg)))

        for point, color, normal in zip(source_points, source_colors, source_normals, strict=False):
            if tree is None:
                self.voxel_map.insert_or_update(point, color, normal, step_index=step_index)
                accepted_points.append(point)
                accepted_colors.append(color)
                continue

            count, indices, distances = tree.search_knn_vector_3d(point, 1)
            if count == 0:
                self.voxel_map.insert_or_update(point, color, normal, step_index=step_index)
                accepted_points.append(point)
                accepted_colors.append(color)
                continue

            nearest_index = int(indices[0])
            distance = float(np.sqrt(distances[0]))
            normal_ok = True
            if len(map_normals) and np.linalg.norm(normal) > 1e-12 and np.linalg.norm(map_normals[nearest_index]) > 1e-12:
                normal_ok = abs(float(np.dot(normal, map_normals[nearest_index]))) >= cos_threshold

            if distance <= self.duplicate_distance and normal_ok:
                self.voxel_map.insert_or_update(point, color, normal, step_index=step_index)
                duplicate_points.append(point)
                duplicate_colors.append(color)
            elif distance <= self.conflict_distance or not normal_ok:
                conflict_points.append(point)
                conflict_colors.append(color)
            else:
                self.voxel_map.insert_or_update(point, color, normal, step_index=step_index)
                accepted_points.append(point)
                accepted_colors.append(color)

        input_count = int(len(source_points))
        accepted_count = int(len(accepted_points))
        duplicate_count = int(len(duplicate_points))
        conflict_count = int(len(conflict_points))
        rejected_count = max(input_count - accepted_count - duplicate_count - conflict_count, 0)
        report = FusionReport(
            accepted_points=accepted_count,
            duplicate_points=duplicate_count,
            conflict_points=conflict_count,
            rejected_points=rejected_count,
            input_points=input_count,
            conflict_ratio=float(conflict_count / max(input_count, 1)),
            duplicate_ratio=float(duplicate_count / max(input_count, 1)),
            accepted_ratio=float(accepted_count / max(input_count, 1)),
            new_frame_names=tuple(new_frame_names),
            params={
                "voxel_size": self.voxel_map.voxel_size,
                "duplicate_distance": self.duplicate_distance,
                "conflict_distance": self.conflict_distance,
                "normal_angle_deg": self.normal_angle_deg,
            },
        )
        return FusionResult(
            fused_cloud=self.voxel_map.to_cloud(),
            report=report,
            debug_clouds=FusionDebugClouds(
                accepted=make_debug_cloud(accepted_points, accepted_colors, (0.1, 0.8, 0.2)),
                duplicate=make_debug_cloud(duplicate_points, duplicate_colors, (0.2, 0.4, 1.0)),
                conflict=make_debug_cloud(conflict_points, conflict_colors, (1.0, 0.1, 0.1)),
            ),
        )

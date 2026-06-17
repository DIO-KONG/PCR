#!/usr/bin/env python3
"""点云预处理工具。

本模块同时提供两种使用方式：
1. 作为 Python API 被配准/建图算法 import。
2. 作为 CLI 独立运行，用于调试单个 PLY 点云。

设计原则：
- 不修改输入点云对象，所有处理函数都返回新对象。
- 地板检测不依赖固定高度阈值，不假设“地板在 y=0.7”。
- 允许使用项目坐标约定：地板法线应对齐到 +Y。
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPROCESS_ROOT = PROJECT_ROOT / "result" / "preprocess"
Y_AXIS = np.asarray([0.0, 1.0, 0.0], dtype=float)


@dataclass
class PointCloudStats:
    """点云基础统计信息。"""

    point_count: int
    bounds_min: list[float]
    bounds_max: list[float]
    extent: list[float]
    center: list[float]
    has_colors: bool
    has_normals: bool


@dataclass
class PlaneCandidate:
    """RANSAC 提取到的单个平面候选。

    `plane_model` 使用 Open3D 约定 `[a, b, c, d]`，表示 `a*x + b*y + c*z + d = 0`。
    这里会把法线方向统一翻到与 +Y 同半球，便于后续计算倾角和旋转。
    """

    plane_model: list[float]
    normal: list[float]
    d: float
    inlier_count: int
    inlier_ratio: float
    tilt_deg_from_y: float
    area_estimate: float
    score: float


@dataclass
class FloorDetectionResult:
    """地板检测结果。

    `floor` 为 None 时表示没有找到可信地板；调用方应跳过地板对齐/去地板。
    """

    status: str
    floor: PlaneCandidate | None
    candidates: list[PlaneCandidate]
    params: dict[str, Any]


@dataclass
class PreprocessResult:
    """预处理结果集合。"""

    cloud: o3d.geometry.PointCloud
    floor_detection: FloorDetectionResult | None = None
    floor_only: o3d.geometry.PointCloud | None = None
    floor_removed: o3d.geometry.PointCloud | None = None
    aligned: o3d.geometry.PointCloud | None = None
    alignment_matrix: np.ndarray | None = None
    stats: PointCloudStats | None = None


def clone_point_cloud(cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """复制点云，避免预处理函数原地修改调用者持有的对象。"""

    return copy.deepcopy(cloud)


def load_point_cloud(path: str | Path) -> o3d.geometry.PointCloud:
    """读取点云并检查是否为空。"""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file does not exist: {path}")
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Point cloud is empty or unreadable: {path}")
    return cloud


def save_point_cloud(path: str | Path, cloud: o3d.geometry.PointCloud) -> None:
    """保存点云，自动创建输出目录。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cloud.is_empty():
        raise ValueError(f"Refuse to write an empty point cloud: {path}")
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f"Open3D failed to write point cloud: {path}")


def describe_point_cloud(cloud: o3d.geometry.PointCloud) -> PointCloudStats:
    """返回点云统计信息。"""

    points = np.asarray(cloud.points)
    if points.size == 0:
        raise ValueError("Cannot describe an empty point cloud.")
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = maxs - mins
    center = (mins + maxs) * 0.5
    return PointCloudStats(
        point_count=int(len(points)),
        bounds_min=mins.astype(float).tolist(),
        bounds_max=maxs.astype(float).tolist(),
        extent=extent.astype(float).tolist(),
        center=center.astype(float).tolist(),
        has_colors=bool(cloud.has_colors()),
        has_normals=bool(cloud.has_normals()),
    )


def print_point_cloud_stats(name: str, stats: PointCloudStats) -> None:
    """以可读格式打印点云统计信息。"""

    print(f"[{name}]")
    print(f"  points: {stats.point_count}")
    print("  bounds min: x={:.4f}, y={:.4f}, z={:.4f}".format(*stats.bounds_min))
    print("  bounds max: x={:.4f}, y={:.4f}, z={:.4f}".format(*stats.bounds_max))
    print("  extent:     x={:.4f}, y={:.4f}, z={:.4f}".format(*stats.extent))
    print("  center:     x={:.4f}, y={:.4f}, z={:.4f}".format(*stats.center))
    print(f"  colors: {stats.has_colors}, normals: {stats.has_normals}")


def voxel_downsample(cloud: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    """体素降采样。

    `voxel_size <= 0` 时直接返回复制后的点云，便于 CLI 中把参数设为 0 跳过该步骤。
    """

    if voxel_size <= 0:
        return clone_point_cloud(cloud)
    return cloud.voxel_down_sample(float(voxel_size))


def statistical_outlier_removal(
    cloud: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """统计离群点过滤。"""

    filtered, _ = cloud.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio),
    )
    return filtered


def radius_outlier_removal(
    cloud: o3d.geometry.PointCloud,
    nb_points: int = 16,
    radius: float = 0.05,
) -> o3d.geometry.PointCloud:
    """半径离群点过滤。"""

    filtered, _ = cloud.remove_radius_outlier(
        nb_points=int(nb_points),
        radius=float(radius),
    )
    return filtered


def estimate_normals(
    cloud: o3d.geometry.PointCloud,
    radius: float = 0.10,
    max_nn: int = 30,
) -> o3d.geometry.PointCloud:
    """估计法线并返回新点云。"""

    result = clone_point_cloud(cloud)
    result.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(radius),
            max_nn=int(max_nn),
        )
    )
    return result


def normalize_plane_model(plane_model: np.ndarray) -> np.ndarray:
    """归一化平面参数，并把法线翻到 +Y 半球。"""

    plane = np.asarray(plane_model, dtype=float).copy()
    normal_norm = float(np.linalg.norm(plane[:3]))
    if normal_norm == 0.0:
        raise ValueError(f"Invalid plane model with zero normal: {plane_model}")
    plane /= normal_norm
    if float(np.dot(plane[:3], Y_AXIS)) < 0.0:
        plane *= -1.0
    return plane


def plane_point_distances(points: np.ndarray, plane_model: np.ndarray) -> np.ndarray:
    """计算点到平面的有符号距离。"""

    plane = normalize_plane_model(plane_model)
    return points @ plane[:3] + float(plane[3])


def plane_area_estimate(points: np.ndarray, normal: np.ndarray) -> float:
    """估计平面内二维包围盒面积。

    该面积只是用于候选排序，不作为严格几何测量。它能帮助区分小块桌面和更大地板区域。
    """

    if len(points) < 3:
        return 0.0

    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    reference = np.asarray([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(reference, normal))) > 0.9:
        reference = np.asarray([0.0, 0.0, 1.0], dtype=float)
    axis_u = np.cross(normal, reference)
    axis_u /= max(float(np.linalg.norm(axis_u)), 1e-12)
    axis_v = np.cross(normal, axis_u)

    projected = np.column_stack([points @ axis_u, points @ axis_v])
    extent = projected.max(axis=0) - projected.min(axis=0)
    return float(max(extent[0], 0.0) * max(extent[1], 0.0))


def make_plane_candidate(
    plane_model: np.ndarray,
    inlier_points: np.ndarray,
    total_points: int,
) -> PlaneCandidate:
    """把 Open3D RANSAC 输出转换成可评分候选。"""

    plane = normalize_plane_model(plane_model)
    normal = plane[:3]
    vertical_alignment = abs(float(np.dot(normal, Y_AXIS)))
    tilt = float(np.degrees(np.arccos(np.clip(vertical_alignment, -1.0, 1.0))))
    inlier_count = int(len(inlier_points))
    inlier_ratio = float(inlier_count / max(total_points, 1))
    area = plane_area_estimate(inlier_points, normal)

    # 分数优先奖励“接近水平”，再看面积和点数支持。
    # 这样不会把 window batch 中巨大的竖直隔板误认为地板。
    score = (vertical_alignment**4) * np.log1p(inlier_count) * np.sqrt(max(area, 1e-9))
    return PlaneCandidate(
        plane_model=plane.astype(float).tolist(),
        normal=normal.astype(float).tolist(),
        d=float(plane[3]),
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        tilt_deg_from_y=tilt,
        area_estimate=area,
        score=float(score),
    )


def extract_planes_iterative(
    cloud: o3d.geometry.PointCloud,
    *,
    max_planes: int = 8,
    distance_threshold: float = 0.03,
    ransac_n: int = 3,
    num_iterations: int = 2000,
    min_plane_points: int = 200,
    min_remaining_ratio: float = 0.20,
) -> list[PlaneCandidate]:
    """迭代提取多个平面候选。

    只取最大平面在办公室场景里很危险：隔板/墙面经常比地板更大。
    因此这里每次提取一个主平面并移除其内点，累计得到多个候选，再由地板检测逻辑筛选。
    """

    working = clone_point_cloud(cloud)
    original_count = len(working.points)
    candidates: list[PlaneCandidate] = []
    for _ in range(int(max_planes)):
        if len(working.points) < int(min_plane_points):
            break
        if len(working.points) / max(original_count, 1) < float(min_remaining_ratio):
            break

        plane_model, inliers = working.segment_plane(
            distance_threshold=float(distance_threshold),
            ransac_n=int(ransac_n),
            num_iterations=int(num_iterations),
        )
        if len(inliers) < int(min_plane_points):
            break

        points = np.asarray(working.points)
        candidate = make_plane_candidate(np.asarray(plane_model), points[inliers], len(points))
        candidates.append(candidate)
        working = working.select_by_index(inliers, invert=True)

    return candidates


def detect_floor_plane(
    cloud: o3d.geometry.PointCloud,
    *,
    plane_voxel_size: float = 0.05,
    max_planes: int = 8,
    distance_threshold: float = 0.03,
    ransac_n: int = 3,
    num_iterations: int = 2000,
    min_plane_points: int = 200,
    max_floor_tilt_deg: float = 35.0,
    min_floor_score: float = 0.1,
) -> FloorDetectionResult:
    """从点云中检测最可信的地板平面。

    这里不使用固定高度阈值。候选筛选只依赖：
    - 平面法线是否接近 +Y/-Y。
    - 平面点数和面积是否足够。
    - 平面综合分数是否超过最低门槛。
    """

    down = voxel_downsample(cloud, plane_voxel_size)
    candidates = extract_planes_iterative(
        down,
        max_planes=max_planes,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        min_plane_points=min_plane_points,
    )

    eligible = [
        item
        for item in candidates
        if item.tilt_deg_from_y <= float(max_floor_tilt_deg)
        and item.inlier_count >= int(min_plane_points)
        and item.score >= float(min_floor_score)
    ]
    floor = max(eligible, key=lambda item: item.score) if eligible else None
    status = "floor_found" if floor is not None else "floor_not_found"
    return FloorDetectionResult(
        status=status,
        floor=floor,
        candidates=candidates,
        params={
            "plane_voxel_size": plane_voxel_size,
            "max_planes": max_planes,
            "distance_threshold": distance_threshold,
            "ransac_n": ransac_n,
            "num_iterations": num_iterations,
            "min_plane_points": min_plane_points,
            "max_floor_tilt_deg": max_floor_tilt_deg,
            "min_floor_score": min_floor_score,
        },
    )


def rotation_matrix_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """计算把 source 单位向量旋转到 target 单位向量的 3x3 矩阵。"""

    source = source / max(float(np.linalg.norm(source)), 1e-12)
    target = target / max(float(np.linalg.norm(target)), 1e-12)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))

    if cross_norm < 1e-12:
        if dot > 0.0:
            return np.eye(3)
        # 180 度旋转的退化情况。正常地板检测会把法线翻到 +Y 半球，很少走到这里。
        return np.diag([1.0, -1.0, -1.0])

    skew = np.asarray(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / (cross_norm**2))


def transform_point_cloud(cloud: o3d.geometry.PointCloud, matrix: np.ndarray) -> o3d.geometry.PointCloud:
    """复制点云并应用 4x4 齐次变换。"""

    result = clone_point_cloud(cloud)
    result.transform(np.asarray(matrix, dtype=float))
    return result


def align_floor_to_y_axis(
    cloud: o3d.geometry.PointCloud,
    floor_plane: PlaneCandidate,
    *,
    rotation_center: str = "centroid",
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    """把地板法线刚性旋转到 +Y。

    默认绕点云 centroid 旋转，避免坐标整体被旋到远离原位置。
    返回对齐后的点云和对应 4x4 变换矩阵。
    """

    normal = np.asarray(floor_plane.normal, dtype=float)
    rotation = rotation_matrix_between_vectors(normal, Y_AXIS)
    points = np.asarray(cloud.points)
    if rotation_center == "origin":
        center = np.zeros(3, dtype=float)
    elif rotation_center == "centroid":
        center = points.mean(axis=0)
    else:
        raise ValueError("rotation_center must be 'centroid' or 'origin'")

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = center - rotation @ center
    return transform_point_cloud(cloud, matrix), matrix


def remove_floor_by_plane(
    cloud: o3d.geometry.PointCloud,
    floor_plane: PlaneCandidate,
    *,
    distance_threshold: float = 0.03,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """按点到地板平面的距离拆分 floor_only 和 floor_removed。

    不使用固定 Y 高度，只使用检测到的平面模型。
    """

    points = np.asarray(cloud.points)
    distances = np.abs(plane_point_distances(points, np.asarray(floor_plane.plane_model)))
    floor_indices = np.flatnonzero(distances <= float(distance_threshold)).tolist()
    floor_only = cloud.select_by_index(floor_indices)
    floor_removed = cloud.select_by_index(floor_indices, invert=True)
    return floor_only, floor_removed


def preprocess_for_visualization(
    cloud: o3d.geometry.PointCloud,
    *,
    voxel_size: float = 0.0,
    sor: bool = False,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """面向可视化的轻量预处理。"""

    result = clone_point_cloud(cloud)
    if sor:
        result = statistical_outlier_removal(result, sor_nb_neighbors, sor_std_ratio)
    if voxel_size > 0:
        result = voxel_downsample(result, voxel_size)
    return result


def preprocess_for_registration(
    cloud: o3d.geometry.PointCloud,
    *,
    align_floor: bool = False,
    remove_floor: bool = False,
    floor_distance_threshold: float = 0.03,
    voxel_size: float = 0.0,
    sor: bool = False,
    sor_nb_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
    radius_outlier: bool = False,
    radius_nb_points: int = 16,
    radius: float = 0.05,
    normals: bool = False,
    normal_radius: float = 0.10,
    normal_max_nn: int = 30,
    floor_kwargs: dict[str, Any] | None = None,
) -> PreprocessResult:
    """面向配准的预处理入口。

    先做可选离群点过滤，再检测/处理地板，最后做可选降采样和法线估计。
    这样地板检测能在较干净的点云上运行，同时输出点云仍可按配准需求降采样。
    """

    result_cloud = clone_point_cloud(cloud)
    if sor:
        result_cloud = statistical_outlier_removal(result_cloud, sor_nb_neighbors, sor_std_ratio)
    if radius_outlier:
        result_cloud = radius_outlier_removal(result_cloud, radius_nb_points, radius)

    detection: FloorDetectionResult | None = None
    floor_only: o3d.geometry.PointCloud | None = None
    floor_removed: o3d.geometry.PointCloud | None = None
    aligned: o3d.geometry.PointCloud | None = None
    alignment_matrix: np.ndarray | None = None

    if align_floor or remove_floor:
        detection = detect_floor_plane(result_cloud, **(floor_kwargs or {}))
        if detection.floor is not None:
            floor_only, floor_removed = remove_floor_by_plane(
                result_cloud,
                detection.floor,
                distance_threshold=floor_distance_threshold,
            )
            if align_floor:
                aligned, alignment_matrix = align_floor_to_y_axis(result_cloud, detection.floor)
                floor_only = transform_point_cloud(floor_only, alignment_matrix) if not floor_only.is_empty() else floor_only
                floor_removed = (
                    transform_point_cloud(floor_removed, alignment_matrix)
                    if not floor_removed.is_empty()
                    else floor_removed
                )
                result_cloud = aligned
            if remove_floor:
                result_cloud = floor_removed

    if voxel_size > 0:
        result_cloud = voxel_downsample(result_cloud, voxel_size)
    if normals:
        result_cloud = estimate_normals(result_cloud, normal_radius, normal_max_nn)

    return PreprocessResult(
        cloud=result_cloud,
        floor_detection=detection,
        floor_only=floor_only,
        floor_removed=floor_removed,
        aligned=aligned,
        alignment_matrix=alignment_matrix,
        stats=describe_point_cloud(result_cloud),
    )


def json_safe(value: Any) -> Any:
    """把 dataclass/numpy/Open3D 无法直接 JSON 化的对象转为普通 Python 类型。"""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_debug_outputs(debug_dir: Path, result: PreprocessResult) -> None:
    """保存地板检测/对齐调试产物。"""

    debug_dir.mkdir(parents=True, exist_ok=True)
    if result.aligned is not None and not result.aligned.is_empty():
        save_point_cloud(debug_dir / "aligned.ply", result.aligned)
    if result.floor_only is not None and not result.floor_only.is_empty():
        save_point_cloud(debug_dir / "floor_only.ply", result.floor_only)
    if result.floor_removed is not None and not result.floor_removed.is_empty():
        save_point_cloud(debug_dir / "floor_removed.ply", result.floor_removed)

    report = {
        "floor_detection": result.floor_detection,
        "alignment_matrix": result.alignment_matrix,
        "output_stats": result.stats,
    }
    (debug_dir / "floor_report.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_floor_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """从 CLI 参数构造地板检测配置。"""

    return {
        "plane_voxel_size": args.plane_voxel_size,
        "max_planes": args.max_planes,
        "distance_threshold": args.floor_distance_threshold,
        "num_iterations": args.plane_iterations,
        "min_plane_points": args.min_plane_points,
        "max_floor_tilt_deg": args.max_floor_tilt_deg,
        "min_floor_score": args.min_floor_score,
    }


def resolve_project_path(path: Path, *, base_dir: Path | None = None) -> Path:
    """解析输出路径。

    约定：
    - 相对路径一律基于项目根目录解析，而不是基于调用者当前 shell 目录。
    - 显式绝对路径保持不变，方便用户有意识地写到外部磁盘或临时目录。
    """

    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    root = PROJECT_ROOT if base_dir is None else base_dir
    return root / expanded


def default_output_path(input_path: Path, args: argparse.Namespace) -> Path:
    """根据输入文件和处理模式生成默认输出路径。"""

    if args.remove_floor:
        suffix = "floor_removed"
    elif args.align_floor:
        suffix = "aligned"
    else:
        suffix = "preprocessed"
    output_root = resolve_project_path(args.output_root)
    return output_root / input_path.stem / f"{suffix}.ply"


def default_debug_dir(input_path: Path, args: argparse.Namespace) -> Path:
    """生成默认 debug 目录。"""

    output_root = resolve_project_path(args.output_root)
    return output_root / input_path.stem / "debug"


def run_cli(args: argparse.Namespace) -> None:
    """CLI 主流程。"""

    input_path = Path(args.input).expanduser().resolve()
    cloud = load_point_cloud(input_path)
    if args.print_stats:
        print_point_cloud_stats("input", describe_point_cloud(cloud))

    result = preprocess_for_registration(
        cloud,
        align_floor=args.align_floor,
        remove_floor=args.remove_floor,
        floor_distance_threshold=args.floor_distance_threshold,
        voxel_size=args.voxel_size,
        sor=args.sor,
        sor_nb_neighbors=args.sor_nb_neighbors,
        sor_std_ratio=args.sor_std_ratio,
        radius_outlier=args.radius_outlier,
        radius_nb_points=args.radius_nb_points,
        radius=args.radius,
        normals=args.estimate_normals,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        floor_kwargs=build_floor_kwargs(args),
    )

    if result.floor_detection is not None:
        print(f"[floor] status: {result.floor_detection.status}")
        if result.floor_detection.floor is not None:
            floor = result.floor_detection.floor
            print(f"  normal: {floor.normal}")
            print(f"  tilt_deg_from_y: {floor.tilt_deg_from_y:.4f}")
            print(f"  inlier_count: {floor.inlier_count}")
            print(f"  inlier_ratio: {floor.inlier_ratio:.4f}")
            print(f"  area_estimate: {floor.area_estimate:.4f}")
            print(f"  score: {floor.score:.4f}")

    output_path: Path | None = None
    if args.output is not None:
        output_path = resolve_project_path(Path(args.output))
    elif args.align_floor or args.remove_floor or args.voxel_size > 0 or args.sor or args.radius_outlier or args.estimate_normals:
        output_path = default_output_path(input_path, args)
    if output_path is not None:
        save_point_cloud(output_path, result.cloud)
        print(f"[output] saved: {output_path}")

    debug_dir: Path | None = None
    if args.write_debug == "__AUTO__":
        debug_dir = default_debug_dir(input_path, args)
    elif args.write_debug is not None:
        debug_dir = resolve_project_path(Path(args.write_debug))
    if debug_dir is not None:
        write_debug_outputs(debug_dir, result)
        print(f"[debug] saved: {debug_dir}")

    if args.print_stats:
        print_point_cloud_stats("output", result.stats or describe_point_cloud(result.cloud))


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Preprocess point clouds for visualization or registration.")
    parser.add_argument("input", help="Input point cloud path, for example .ply/.pcd/.xyz.")
    parser.add_argument("--output", type=Path, default=None, help="Output point cloud path.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_PREPROCESS_ROOT,
        help="Base directory for default outputs. Relative paths are resolved from the project root.",
    )
    parser.add_argument("--print-stats", action="store_true", help="Print input and output statistics.")

    parser.add_argument("--voxel-size", type=float, default=0.0, help="Final voxel downsample size. 0 disables it.")
    parser.add_argument("--sor", action="store_true", help="Enable statistical outlier removal.")
    parser.add_argument("--sor-nb-neighbors", type=int, default=20)
    parser.add_argument("--sor-std-ratio", type=float, default=2.0)
    parser.add_argument("--radius-outlier", action="store_true", help="Enable radius outlier removal.")
    parser.add_argument("--radius-nb-points", type=int, default=16)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--estimate-normals", action="store_true")
    parser.add_argument("--normal-radius", type=float, default=0.10)
    parser.add_argument("--normal-max-nn", type=int, default=30)

    parser.add_argument("--align-floor", action="store_true", help="Rotate detected floor normal to +Y.")
    parser.add_argument("--remove-floor", action="store_true", help="Remove points close to detected floor plane.")
    parser.add_argument(
        "--write-debug",
        nargs="?",
        const="__AUTO__",
        default=None,
        help="Directory for floor debug artifacts. Omit value to use result/preprocess/<stem>/debug.",
    )
    parser.add_argument("--plane-voxel-size", type=float, default=0.05)
    parser.add_argument("--floor-distance-threshold", type=float, default=0.03)
    parser.add_argument("--max-planes", type=int, default=8)
    parser.add_argument("--plane-iterations", type=int, default=2000)
    parser.add_argument("--min-plane-points", type=int, default=200)
    parser.add_argument("--max-floor-tilt-deg", type=float, default=35.0)
    parser.add_argument("--min-floor-score", type=float, default=0.1)
    args = parser.parse_args()

    if args.voxel_size < 0:
        parser.error("--voxel-size must be >= 0")
    if args.plane_voxel_size <= 0:
        parser.error("--plane-voxel-size must be positive")
    if args.floor_distance_threshold <= 0:
        parser.error("--floor-distance-threshold must be positive")
    if args.max_planes < 1:
        parser.error("--max-planes must be >= 1")
    if args.min_plane_points < 3:
        parser.error("--min-plane-points must be >= 3")
    if args.max_floor_tilt_deg < 0 or args.max_floor_tilt_deg > 90:
        parser.error("--max-floor-tilt-deg must be in [0, 90]")
    return args


if __name__ == "__main__":
    run_cli(parse_args())

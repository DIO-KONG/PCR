#!/usr/bin/env python3
"""使用 Open3D 可视化一个或两个点云文件。

典型用法：

    .env/bin/python utils/visualize.py cloud.ply
    .env/bin/python utils/visualize.py target.ply source.ply

双点云模式主要用于配准检查：默认把第一个点云画成灰色，第二个点云画成黄色，
并添加 XYZ 坐标轴。脚本只做可视化，不修改输入文件。
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import open3d as o3d


DEFAULT_TARGET_COLOR = (0.55, 0.55, 0.55)
DEFAULT_SOURCE_COLOR = (1.0, 0.82, 0.05)


def parse_color(value: str) -> tuple[float, float, float]:
    """解析命令行颜色参数。

    支持 `r,g,b` 格式，数值范围为 0..1，例如 `1,0,0` 表示红色。
    """

    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be formatted as r,g,b")
    try:
        color = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("color values must be numbers") from exc
    if any(component < 0.0 or component > 1.0 for component in color):
        raise argparse.ArgumentTypeError("color values must be in [0, 1]")
    return color  # type: ignore[return-value]


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """读取点云并检查点数。

    Open3D 对不存在或无法解析的文件常常只返回空点云，因此这里显式报错，
    避免打开一个空窗口后难以判断原因。
    """

    if not path.exists():
        raise FileNotFoundError(f"Point cloud file does not exist: {path}")
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"Point cloud is empty or unreadable: {path}")
    return cloud


def describe_point_cloud(name: str, path: Path, cloud: o3d.geometry.PointCloud) -> None:
    """打印点云统计信息，方便快速判断尺度、范围和颜色状态。"""

    points = np.asarray(cloud.points)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = maxs - mins
    center = (mins + maxs) * 0.5
    has_colors = cloud.has_colors()
    has_normals = cloud.has_normals()
    print(f"[{name}] {path}")
    print(f"  points: {len(points)}")
    print(f"  bounds min: x={mins[0]:.4f}, y={mins[1]:.4f}, z={mins[2]:.4f}")
    print(f"  bounds max: x={maxs[0]:.4f}, y={maxs[1]:.4f}, z={maxs[2]:.4f}")
    print(f"  extent:     x={extent[0]:.4f}, y={extent[1]:.4f}, z={extent[2]:.4f}")
    print(f"  center:     x={center[0]:.4f}, y={center[1]:.4f}, z={center[2]:.4f}")
    print(f"  colors: {has_colors}, normals: {has_normals}")


def make_axis(clouds: list[o3d.geometry.PointCloud], axis_size: float | None) -> o3d.geometry.TriangleMesh:
    """创建 XYZ 坐标轴。

    如果用户没有指定坐标轴大小，则根据所有点云包围盒的最长边自动估计，
    避免坐标轴相对点云过大或过小。
    """

    if axis_size is None:
        extents = []
        for cloud in clouds:
            bbox = cloud.get_axis_aligned_bounding_box()
            extents.append(np.asarray(bbox.get_extent(), dtype=float))
        max_extent = float(np.max(np.vstack(extents)))
        axis_size = max(max_extent * 0.15, 0.25)
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axis_size), origin=(0.0, 0.0, 0.0))


def prepare_for_display(
    cloud: o3d.geometry.PointCloud,
    *,
    color: tuple[float, float, float],
    recolor: bool,
) -> o3d.geometry.PointCloud:
    """复制点云并按需上色。

    可视化时不直接修改原始对象，保持函数行为单纯，避免未来复用时误写输入。
    """

    display_cloud = copy.deepcopy(cloud)
    if recolor:
        display_cloud.paint_uniform_color(color)
    return display_cloud


def visualize(args: argparse.Namespace) -> None:
    """主流程：加载点云、打印统计、构建几何体并打开 Open3D 窗口。"""

    paths = [Path(path).expanduser().resolve() for path in args.pointclouds]
    clouds = [load_point_cloud(path) for path in paths]

    for index, (path, cloud) in enumerate(zip(paths, clouds, strict=True), 1):
        describe_point_cloud(f"cloud_{index}", path, cloud)

    recolor = not args.keep_colors
    geometries: list[o3d.geometry.Geometry] = []
    if len(clouds) == 1:
        geometries.append(
            prepare_for_display(
                clouds[0],
                color=args.source_color,
                recolor=args.recolor_single and recolor,
            )
        )
    else:
        geometries.append(prepare_for_display(clouds[0], color=args.target_color, recolor=recolor))
        geometries.append(prepare_for_display(clouds[1], color=args.source_color, recolor=recolor))

    if not args.no_axis:
        geometries.append(make_axis(clouds, args.axis_size))

    title = args.window_name
    if title is None:
        title = "Open3D Point Cloud Viewer" if len(paths) == 1 else "Open3D Point Cloud Registration Viewer"
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=args.width,
        height=args.height,
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="Visualize one or two point cloud files with Open3D.",
    )
    parser.add_argument(
        "pointclouds",
        nargs="+",
        help="One or two point cloud paths, for example .ply/.pcd/.xyz.",
    )
    parser.add_argument("--keep-colors", action="store_true", help="Keep original point colors.")
    parser.add_argument(
        "--recolor-single",
        action="store_true",
        help="Paint a single input cloud with --source-color. By default single-cloud colors are preserved.",
    )
    parser.add_argument("--target-color", type=parse_color, default=DEFAULT_TARGET_COLOR)
    parser.add_argument("--source-color", type=parse_color, default=DEFAULT_SOURCE_COLOR)
    parser.add_argument("--axis-size", type=float, default=None, help="Coordinate frame size. Auto if omitted.")
    parser.add_argument("--no-axis", action="store_true", help="Do not draw XYZ coordinate frame.")
    parser.add_argument("--window-name", default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    if len(args.pointclouds) not in (1, 2):
        parser.error("expected one or two point cloud paths")
    if args.axis_size is not None and args.axis_size <= 0:
        parser.error("--axis-size must be positive")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    return args


if __name__ == "__main__":
    visualize(parse_args())

#!/usr/bin/env python3
"""运行 Depth Anything 3 batch，并导出每个 batch 的预测与融合点云。

这个脚本只负责“从 RGB 图像生成 DA3 局部 batch 结果”，不负责跨 batch 配准。
跨 batch 配准和建图实验由 `pcr/` 下的 pipeline 驱动。
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_IMAGE_DIR = Path("da3/data/raw/image")
DEFAULT_OUTPUT_DIR = Path("da3/data/raw/pointcloud")
FRAME_RE = re.compile(r"^(\d+)(?:-rgb)?\.png$")


@dataclass(frozen=True)
class BatchSpec:
    """一个 DA3 推理 batch 的定义。

    `frames` 使用数字帧号，输出文件名由 `stem` 统一生成，避免字符串排序造成 10 排在 2 前面。
    """

    index: int
    label: str
    frames: tuple[int, ...]

    @property
    def stem(self) -> str:
        first = self.frames[0]
        last = self.frames[-1]
        return f"batch_{self.index:03d}_{self.label}_{first:03d}-{last:03d}"


def build_batches(
    *,
    frame_start: int,
    frame_end: int,
    baseline_size: int,
    window_size: int,
) -> list[BatchSpec]:
    """构造 DA3 batch 列表。

    baseline 使用 `frame_start..frame_start+baseline_size-1` 初始化地图。
    后续 window 使用“当前帧和前 `window_size-1` 帧”，例如 5 图 window
    在第 13 帧时为 `9..13`。
    """

    if frame_end < frame_start:
        raise ValueError(f"frame_end must be >= frame_start, got {frame_start}..{frame_end}")
    if baseline_size < 1:
        raise ValueError("--baseline-size must be >= 1")
    if window_size < 2:
        raise ValueError("--window-size must be >= 2")
    baseline_last = frame_start + baseline_size - 1
    if baseline_last > frame_end:
        raise ValueError("Baseline range exceeds requested frame range.")

    batches = [BatchSpec(1, "baseline", tuple(range(frame_start, baseline_last + 1)))]
    batch_index = 2
    for current_frame in range(baseline_last + 1, frame_end + 1):
        first = current_frame - window_size + 1
        if first < frame_start:
            raise ValueError(f"Window {first}..{current_frame} starts before frame_start={frame_start}")
        batches.append(BatchSpec(batch_index, "window", tuple(range(first, current_frame + 1))))
        batch_index += 1
    return batches


def discover_images(image_dir: Path, *, frame_start: int, frame_end: int) -> dict[int, Path]:
    """按数字前缀发现输入图像，并校验请求范围完整。

    支持 `1.png` 和历史 `1-rgb.png` 两种命名。若同一帧同时存在两种文件，
    优先报错，避免 DA3 batch 输入不稳定。
    """

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    frames: dict[int, Path] = {}
    for path in image_dir.iterdir():
        match = FRAME_RE.match(path.name)
        if match is None:
            continue
        frame_id = int(match.group(1))
        if frame_id in frames:
            raise ValueError(f"Duplicate frame id {frame_id}: {frames[frame_id]} and {path}")
        frames[frame_id] = path

    expected = set(range(frame_start, frame_end + 1))
    actual = set(frames)
    if not expected.issubset(actual):
        missing = sorted(expected - actual)
        raise ValueError(f"Missing requested frames {frame_start}..{frame_end}: {missing}")

    return {frame_id: frames[frame_id] for frame_id in sorted(expected)}


def require_cuda() -> None:
    """检查 CUDA。

    默认 Nested 模型较重，项目约定不静默退回 CPU，避免长时间运行后得到低效或不一致结果。
    """

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for depth-anything/DA3NESTED-GIANT-LARGE-1.1, "
            "but torch.cuda.is_available() is False."
        )


def load_model(model_name: str):
    """加载 DA3 模型到 CUDA。"""

    import torch
    from depth_anything_3.api import DepthAnything3

    require_cuda()
    torch.backends.cuda.matmul.allow_tf32 = True
    model = DepthAnything3.from_pretrained(model_name).to("cuda")
    model.eval()
    return model


def extrinsics_to_4x4(extrinsics: np.ndarray) -> np.ndarray:
    """把 DA3 外参统一成 4x4 齐次矩阵。

    DA3 外参是 world-to-camera；生成点云时会取逆，把相机坐标点变换到 batch 局部 world。
    """

    exts = np.asarray(extrinsics, dtype=np.float64)
    if exts.ndim != 3:
        raise ValueError(f"Expected extrinsics with 3 dims, got shape {exts.shape}")
    if exts.shape[1:] == (4, 4):
        return exts
    if exts.shape[1:] == (3, 4):
        bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (exts.shape[0], 1, 4))
        return np.concatenate([exts, bottom], axis=1)
    raise ValueError(f"Expected extrinsics shaped (N, 3, 4) or (N, 4, 4), got {exts.shape}")


def prediction_to_arrays(prediction) -> dict[str, np.ndarray]:
    """把 DA3 Prediction 对象转换成 numpy 字典，便于保存 NPZ 和生成点云。"""

    arrays = {
        "depth": np.asarray(prediction.depth),
        "extrinsics": extrinsics_to_4x4(np.asarray(prediction.extrinsics)),
        "intrinsics": np.asarray(prediction.intrinsics),
        "processed_images": np.asarray(prediction.processed_images),
    }
    conf = getattr(prediction, "conf", None)
    if conf is not None:
        arrays["conf"] = np.asarray(conf)
    return arrays


def save_prediction_npz(
    output_path: Path,
    image_paths: Iterable[Path],
    arrays: dict[str, np.ndarray],
) -> None:
    """保存 batch 的深度、置信度、内参、外参和处理后图像。"""

    payload = {
        "image_names": np.asarray([path.name for path in image_paths]),
        "depth": arrays["depth"],
        "extrinsics": arrays["extrinsics"],
        "intrinsics": arrays["intrinsics"],
        "processed_images": arrays["processed_images"],
    }
    if "conf" in arrays:
        payload["conf"] = arrays["conf"]
    np.savez_compressed(output_path, **payload)


def depth_to_world_points(
    depth: np.ndarray,
    image: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
    conf: np.ndarray | None,
    conf_percentile: float,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把单张深度图反投影成 batch 局部 world 坐标系下的 3D 点。

    反投影流程：
    1. 用内参把像素 + depth 转成 camera 坐标。
    2. 取 DA3 world-to-camera 外参的逆，得到 camera-to-world。
    3. 把 camera 点变换到 batch 局部 world。
    """

    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got {depth.shape}")
    if image.shape[:2] != depth.shape:
        raise ValueError(f"Image shape {image.shape[:2]} does not match depth shape {depth.shape}")

    valid = np.isfinite(depth) & (depth > 0)
    if conf is not None:
        threshold = np.percentile(conf[np.isfinite(conf)], conf_percentile)
        valid &= np.isfinite(conf) & (conf >= threshold)
    if stride > 1:
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[::stride, ::stride] = True
        valid &= stride_mask

    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    z = depth[rows, cols].astype(np.float64)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise ValueError(f"Invalid intrinsic matrix with zero focal length: {intrinsic}")

    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    cam_points = np.column_stack([x, y, z, np.ones_like(z)])
    world_from_camera = np.linalg.inv(extrinsic_w2c)
    world_points = (world_from_camera @ cam_points.T).T[:, :3]
    colors = image[rows, cols].astype(np.float64) / 255.0
    return world_points, colors


def make_point_cloud(
    arrays: dict[str, np.ndarray],
    conf_percentile: float,
    stride: int,
    voxel_size: float,
) -> tuple[object, int, int]:
    """把一个 batch 的多视角深度融合成 Open3D 点云。

    当前融合只发生在 DA3 batch 内部。跨 batch 的重影过滤和建图由 testbench 处理。
    """

    import open3d as o3d

    depth = arrays["depth"]
    images = arrays["processed_images"]
    intrinsics = arrays["intrinsics"]
    extrinsics = arrays["extrinsics"]
    conf = arrays.get("conf")

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for view_index in range(depth.shape[0]):
        view_conf = None if conf is None else conf[view_index]
        points, colors = depth_to_world_points(
            depth=depth[view_index],
            image=images[view_index],
            intrinsic=intrinsics[view_index],
            extrinsic_w2c=extrinsics[view_index],
            conf=view_conf,
            conf_percentile=conf_percentile,
            stride=stride,
        )
        if points.size:
            all_points.append(points)
            all_colors.append(colors)

    if not all_points:
        raise RuntimeError("No valid 3D points survived depth/confidence filtering.")

    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(merged_points)
    cloud.colors = o3d.utility.Vector3dVector(merged_colors)
    before = len(cloud.points)
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size=voxel_size)
    after = len(cloud.points)
    return cloud, before, after


def run(args: argparse.Namespace) -> None:
    """脚本主流程。"""

    image_dir = args.image_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = discover_images(image_dir, frame_start=args.frame_start, frame_end=args.frame_end)
    batches = build_batches(
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        baseline_size=args.baseline_size,
        window_size=args.window_size,
    )

    print(f"Discovered {len(frames)} requested frames: {min(frames)}..{max(frames)}")
    print(f"Configured {len(batches)} batches:")
    preview = batches[:3] + ([] if len(batches) <= 6 else [BatchSpec(0, "...", tuple())]) + batches[-3:]
    for batch in preview:
        if batch.label == "...":
            print("  ...")
        else:
            print(f"  {batch.stem}: {list(batch.frames)}")

    if args.dry_run:
        return

    model = load_model(args.model)

    import open3d as o3d

    for batch in batches:
        image_paths = [frames[frame] for frame in batch.frames]
        npz_path = output_dir / f"{batch.stem}.npz"
        ply_path = output_dir / f"{batch.stem}.ply"
        if args.skip_existing and npz_path.exists() and (args.npz_only or ply_path.exists()):
            print(f"\nSkipping existing {batch.stem}")
            continue

        print(f"\nRunning {batch.stem} with {len(image_paths)} images")
        prediction = model.inference(
            image=[str(path) for path in image_paths],
            use_ray_pose=True,
            ref_view_strategy="middle",
            process_res=args.process_res,
            process_res_method=args.process_res_method,
        )
        arrays = prediction_to_arrays(prediction)

        save_prediction_npz(npz_path, image_paths, arrays)
        print(f"  saved prediction: {npz_path}")

        if args.npz_only:
            continue
        cloud, before, after = make_point_cloud(
            arrays=arrays,
            conf_percentile=args.conf_percentile,
            stride=args.point_stride,
            voxel_size=args.voxel_size,
        )
        if not o3d.io.write_point_cloud(str(ply_path), cloud):
            raise RuntimeError(f"Open3D failed to write point cloud: {ply_path}")

        print(f"  saved pointcloud: {ply_path}")
        print(f"  points: {before} before downsample, {after} after downsample")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=327)
    parser.add_argument("--baseline-size", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--conf-percentile", type=float, default=20.0)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--skip-existing", action="store_true", help="Skip batches whose outputs already exist.")
    parser.add_argument("--npz-only", action="store_true", help="Only save DA3 predictions, skip fused batch PLY output.")
    parser.add_argument("--dry-run", action="store_true", help="Validate image discovery and batches only.")
    args = parser.parse_args()
    if args.point_stride < 1:
        parser.error("--point-stride must be >= 1")
    if not 0.0 <= args.conf_percentile <= 100.0:
        parser.error("--conf-percentile must be in [0, 100]")
    return args


if __name__ == "__main__":
    run(parse_args())

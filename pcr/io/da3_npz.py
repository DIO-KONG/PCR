from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


def load_da3_batch(path: str | Path) -> dict[str, np.ndarray]:
    """读取 DA3 batch NPZ，并转成普通 dict。

    `np.load` 返回的对象依赖文件句柄生命周期。这里立即复制成普通 dict，
    方便算法模块安全传递。
    """

    data = np.load(Path(path), allow_pickle=False)
    return {key: np.asarray(data[key]) for key in data.files}


def image_names(batch: dict[str, np.ndarray]) -> list[str]:
    """返回 DA3 batch 中的图像名列表。"""

    return [str(item) for item in batch["image_names"]]


def confidence_mask(conf: np.ndarray | None, percentile: float) -> np.ndarray | None:
    """按 DA3 confidence percentile 生成 mask。

    `percentile=20` 表示丢弃最低 20% confidence，保留更可信的 80%。
    """

    if conf is None:
        return None
    finite = conf[np.isfinite(conf)]
    if finite.size == 0:
        return np.zeros_like(conf, dtype=bool)
    threshold = np.percentile(finite, float(percentile))
    return np.isfinite(conf) & (conf >= threshold)


def frame_to_point_cloud(
    batch: dict[str, np.ndarray],
    frame_name: str,
    *,
    conf_percentile: float = 20.0,
    stride: int = 1,
    max_points: int | None = None,
    random_seed: int = 7,
) -> o3d.geometry.PointCloud:
    """从 DA3 NPZ 中把单个 frame 反投影为 batch 局部点云。

    输出坐标仍是 DA3 batch 原始局部 world。若后续配准使用的是
    floor-removed batch 点云，调用方必须再应用该 batch 的预处理矩阵和地板过滤。
    """

    names = image_names(batch)
    if frame_name not in names:
        raise ValueError(f"Frame {frame_name!r} does not exist in batch: {names}")
    index = names.index(frame_name)
    depth = np.asarray(batch["depth"][index])
    image = np.asarray(batch.get("processed_images")[index]) if "processed_images" in batch else None
    intrinsic = np.asarray(batch["intrinsics"][index], dtype=np.float64)
    extrinsic_w2c = np.asarray(batch["extrinsics"][index], dtype=np.float64)
    conf = None if "conf" not in batch else np.asarray(batch["conf"][index])

    valid = np.isfinite(depth) & (depth > 0)
    mask = confidence_mask(conf, conf_percentile)
    if mask is not None:
        valid &= mask
    if stride > 1:
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[:: int(stride), :: int(stride)] = True
        valid &= stride_mask

    rows, cols = np.nonzero(valid)
    if max_points is not None and rows.size > int(max_points):
        rng = np.random.default_rng(int(random_seed) + index * 1009)
        selected = rng.choice(rows.size, size=int(max_points), replace=False)
        rows = rows[selected]
        cols = cols[selected]

    cloud = o3d.geometry.PointCloud()
    if rows.size == 0:
        return cloud

    z = depth[rows, cols].astype(np.float64)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise ValueError(f"Invalid intrinsic with zero focal length for frame {frame_name}: {intrinsic}")

    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    camera_points = np.column_stack([x, y, z, np.ones_like(z)])
    camera_to_world = np.linalg.inv(extrinsic_w2c)
    points = (camera_to_world @ camera_points.T).T[:, :3]
    cloud.points = o3d.utility.Vector3dVector(points)

    if image is not None:
        colors = image[rows, cols].astype(np.float64)
        if colors.size and float(np.max(colors)) > 1.0:
            colors = colors / 255.0
        cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    else:
        cloud.paint_uniform_color((0.8, 0.8, 0.8))
    return cloud

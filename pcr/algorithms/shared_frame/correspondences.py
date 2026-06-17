from __future__ import annotations

from typing import Any

import numpy as np

from pcr.io.da3_npz import image_names


def valid_conf_mask(conf: np.ndarray | None, percentile: float) -> np.ndarray | None:
    """根据 confidence percentile 生成高置信 mask。

    `percentile=20` 表示丢弃最低 20% confidence，保留较高的 80%。
    """

    if conf is None:
        return None
    finite = conf[np.isfinite(conf)]
    if finite.size == 0:
        return np.zeros_like(conf, dtype=bool)
    threshold = np.percentile(finite, float(percentile))
    return np.isfinite(conf) & (conf >= threshold)


def pixels_to_world(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """把像素深度反投影到 DA3 batch 局部 world 坐标。

    DA3 的 extrinsics 是 world-to-camera，因此这里取逆得到 camera-to-world。
    """

    z = depth[rows, cols].astype(np.float64)
    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if fx == 0.0 or fy == 0.0:
        raise ValueError(f"Invalid intrinsic with zero focal length: {intrinsic}")

    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    camera_points = np.column_stack([x, y, z, np.ones_like(z)])
    camera_to_world = np.linalg.inv(extrinsic_w2c)
    return (camera_to_world @ camera_points.T).T[:, :3]


def sample_shared_frame_points(
    source_batch: dict[str, np.ndarray],
    target_batch: dict[str, np.ndarray],
    source_index: int,
    target_index: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """从一张共享帧中提取 source->target 的同像素 3D 对应点。"""

    depth_s = source_batch["depth"][source_index]
    depth_t = target_batch["depth"][target_index]
    if depth_s.shape != depth_t.shape:
        raise ValueError(f"Shared frame depth shapes differ: source={depth_s.shape}, target={depth_t.shape}")

    sampling = config.get("sampling", {})
    stride = int(sampling.get("stride", 6))
    conf_percentile = float(sampling.get("conf_percentile", 20.0))
    max_points = int(sampling.get("max_points_per_frame", 8000))
    random_seed = int(sampling.get("random_seed", 7))

    valid = np.isfinite(depth_s) & np.isfinite(depth_t) & (depth_s > 0.0) & (depth_t > 0.0)
    conf_s = source_batch.get("conf")
    conf_t = target_batch.get("conf")
    mask_s = None if conf_s is None else valid_conf_mask(conf_s[source_index], conf_percentile)
    mask_t = None if conf_t is None else valid_conf_mask(conf_t[target_index], conf_percentile)
    if mask_s is not None:
        valid &= mask_s
    if mask_t is not None:
        valid &= mask_t

    if stride > 1:
        stride_mask = np.zeros_like(valid, dtype=bool)
        stride_mask[::stride, ::stride] = True
        valid &= stride_mask

    rows, cols = np.nonzero(valid)
    if rows.size > max_points:
        # 固定随机种子保证 sweep 与正式 pipeline 结果可复现。
        rng = np.random.default_rng(random_seed + int(source_index) * 1009 + int(target_index))
        indices = rng.choice(rows.size, size=max_points, replace=False)
        rows = rows[indices]
        cols = cols[indices]

    source_points = pixels_to_world(
        depth_s,
        source_batch["intrinsics"][source_index],
        source_batch["extrinsics"][source_index],
        rows,
        cols,
    )
    target_points = pixels_to_world(
        depth_t,
        target_batch["intrinsics"][target_index],
        target_batch["extrinsics"][target_index],
        rows,
        cols,
    )
    return source_points, target_points


def collect_correspondences(
    source_batch: dict[str, np.ndarray],
    target_batch: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """收集所有共享帧的像素级 3D 对应关系。"""

    source_names = image_names(source_batch)
    target_names = image_names(target_batch)
    configured = config.get("shared_frames")
    if configured:
        shared_frames = [str(item) for item in configured]
    else:
        shared_frames = sorted(set(source_names) & set(target_names))

    source_all: list[np.ndarray] = []
    target_all: list[np.ndarray] = []
    frame_all: list[np.ndarray] = []
    frame_stats: list[dict[str, Any]] = []
    for name in shared_frames:
        if name not in source_names or name not in target_names:
            raise ValueError(f"Configured shared frame is not in both batches: {name}")
        source_index = source_names.index(name)
        target_index = target_names.index(name)
        source_points, target_points = sample_shared_frame_points(
            source_batch,
            target_batch,
            source_index,
            target_index,
            config,
        )
        frame_stats.append(
            {
                "frame": name,
                "source_index": source_index,
                "target_index": target_index,
                "correspondence_count": int(len(source_points)),
            }
        )
        if len(source_points):
            source_all.append(source_points)
            target_all.append(target_points)
            frame_all.append(np.full(len(source_points), name, dtype=object))

    if not source_all:
        raise RuntimeError("No valid shared-frame correspondences were collected.")
    return np.vstack(source_all), np.vstack(target_all), np.concatenate(frame_all), frame_stats


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from pcr.domain import BatchRef
from pcr.io.da3_npz import frame_to_point_cloud, load_da3_batch
from pcr.preprocessing.preprocess import PlaneCandidate, remove_floor_by_plane, transform_point_cloud


def load_floor_report(batch: BatchRef) -> dict[str, Any]:
    """读取 batch 预处理调试报告。

    新增帧点云必须复用所属 batch 的地板平面和对齐矩阵。若单帧重新检测地板，
    它会落入另一个坐标系，后续 `T_window_to_submap` 就不再适用。
    """

    report_path = Path(batch.preprocessed_cloud_path).parent / "debug" / "floor_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Missing floor report for {batch.batch_id}: {report_path}. "
            "Run PreprocessService.ensure_floor_removed first."
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


def apply_batch_preprocess_to_frame(
    cloud: o3d.geometry.PointCloud,
    report: dict[str, Any],
    *,
    floor_distance_threshold: float = 0.03,
    remove_floor: bool = True,
) -> o3d.geometry.PointCloud:
    """把原始 DA3 单帧点云转换到 batch 预处理坐标系。

    处理顺序必须与 batch PLY 的预处理保持一致：floor-removed 模式先按原始
    batch 坐标中的地板平面删除地板，再应用 floor alignment matrix；align-only
    模式只应用同一个 alignment matrix，保留地板点参与融合。
    """

    result = o3d.geometry.PointCloud(cloud)
    floor_payload = report.get("floor_detection", {}).get("floor")
    if remove_floor and floor_payload:
        _, result = remove_floor_by_plane(
            result,
            PlaneCandidate(**floor_payload),
            distance_threshold=float(floor_distance_threshold),
        )

    matrix_payload = report.get("alignment_matrix")
    if matrix_payload is not None and not result.is_empty():
        result = transform_point_cloud(result, np.asarray(matrix_payload, dtype=float))
    return result


def build_new_frame_clouds(
    source: BatchRef,
    target: BatchRef,
    *,
    conf_percentile: float,
    stride: int,
    max_points_per_frame: int | None,
    random_seed: int,
    floor_distance_threshold: float = 0.03,
    remove_floor: bool = True,
) -> tuple[o3d.geometry.PointCloud, tuple[str, ...]]:
    """从 source batch 中构造“非共享新增帧”的预处理点云。

    对 5 图 window，相邻 target 通常共享前 4 帧；本函数自动用 image name
    差集找新增帧，因此也兼容 baseline/window 命名从 `9-rgb.png` 切到 `9.png`。
    """

    source_batch = load_da3_batch(source.npz_path)
    target_batch = load_da3_batch(target.npz_path)
    source_names = [str(item) for item in source_batch["image_names"]]
    target_names = {str(item) for item in target_batch["image_names"]}
    new_frame_names = tuple(name for name in source_names if name not in target_names)
    if not new_frame_names:
        raise RuntimeError(f"No non-shared new frames found for {source.batch_id} vs {target.batch_id}.")

    report = load_floor_report(source)
    merged = o3d.geometry.PointCloud()
    for frame_name in new_frame_names:
        raw_cloud = frame_to_point_cloud(
            source_batch,
            frame_name,
            conf_percentile=float(conf_percentile),
            stride=int(stride),
            max_points=max_points_per_frame,
            random_seed=int(random_seed),
        )
        if raw_cloud.is_empty():
            continue
        processed = apply_batch_preprocess_to_frame(
            raw_cloud,
            report,
            floor_distance_threshold=floor_distance_threshold,
            remove_floor=remove_floor,
        )
        if not processed.is_empty():
            merged += processed

    if merged.is_empty():
        raise RuntimeError(f"All new-frame points were filtered out for {source.batch_id}: {new_frame_names}")
    return merged, new_frame_names

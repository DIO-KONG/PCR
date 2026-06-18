from __future__ import annotations

from pathlib import Path

from pcr.domain import BatchRef
from pcr.io.pointcloud_io import load_point_cloud
from pcr.preprocessing.preprocess import (
    load_point_cloud as load_raw_point_cloud,
    preprocess_for_registration,
    save_point_cloud,
    write_debug_outputs,
)


class PreprocessService:
    """确保配准需要的去地板点云存在。

    当前复用 `pcr.preprocessing.preprocess` 的成熟实现；本类只负责把“缺失则补齐”从
    walk-forward runner 中拿出来，避免 runner 同时承担预处理细节。
    """

    def ensure_floor_removed(self, batch: BatchRef) -> Path:
        output_path = Path(batch.preprocessed_cloud_path)
        if output_path.exists():
            cloud = load_point_cloud(output_path)
            if not cloud.is_empty():
                return output_path

        print(f"[preprocess] {batch.raw_cloud_path} -> {output_path}", flush=True)
        cloud = load_raw_point_cloud(batch.raw_cloud_path)
        result = preprocess_for_registration(cloud, align_floor=True, remove_floor=True)
        save_point_cloud(output_path, result.cloud)
        write_debug_outputs(output_path.parent / "debug", result)
        return output_path

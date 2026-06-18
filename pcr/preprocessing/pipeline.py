from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pcr.domain import BatchRef
from pcr.io.pointcloud_io import load_point_cloud
from pcr.preprocessing.preprocess import (
    load_point_cloud as load_raw_point_cloud,
    preprocess_for_registration,
    save_point_cloud,
    write_debug_outputs,
)


class PreprocessService:
    """确保配准需要的预处理点云存在。

    当前复用 `pcr.preprocessing.preprocess` 的成熟实现；本类只负责把“缺失则补齐”从
    walk-forward runner 中拿出来，避免 runner 同时承担预处理细节。
    """

    def __init__(self, *, align_floor: bool = True, remove_floor: bool = True) -> None:
        self.align_floor = bool(align_floor)
        self.remove_floor = bool(remove_floor)

    @classmethod
    def from_config(cls, config: dict | None) -> "PreprocessService":
        """从实验配置构造预处理服务。

        默认保持历史行为：地板对齐后去地板。对照实验可设置
        `preprocess.remove_floor: false`，只修正地板方向、不删除地板点。
        """

        config = config or {}
        return cls(
            align_floor=bool(config.get("align_floor", True)),
            remove_floor=bool(config.get("remove_floor", True)),
        )

    def ensure_floor_removed(self, batch: BatchRef) -> Path:
        output_path = Path(batch.preprocessed_cloud_path)
        if output_path.exists():
            cloud = load_point_cloud(output_path)
            if not cloud.is_empty():
                return output_path

        print(f"[preprocess] {batch.raw_cloud_path} -> {output_path}", flush=True)
        cloud = load_raw_point_cloud(batch.raw_cloud_path)
        result = preprocess_for_registration(
            cloud,
            align_floor=self.align_floor,
            remove_floor=self.remove_floor,
        )
        save_point_cloud(output_path, result.cloud)
        write_debug_outputs(output_path.parent / "debug", result)
        return output_path

    def alignment_matrix(self, batch: BatchRef) -> np.ndarray:
        """读取 raw 点云到预处理点云坐标系的地板对齐矩阵。

        shared-frame coarse 是在 DA3 NPZ 的 raw batch 坐标里估计的，而 ICP 使用
        的 PLY 可能已经做过 floor alignment。进入 ICP 前必须把矩阵从 raw
        坐标系转换到 preprocessed 坐标系；若未启用对齐，则该矩阵为单位阵。
        """

        self.ensure_floor_removed(batch)
        report_path = Path(batch.preprocessed_cloud_path).parent / "debug" / "floor_report.json"
        if not report_path.exists():
            return np.eye(4, dtype=float)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        matrix = report.get("alignment_matrix")
        if matrix is None:
            return np.eye(4, dtype=float)
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"Invalid alignment_matrix shape in {report_path}: {matrix.shape}")
        return matrix

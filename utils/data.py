from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.pointcloud import point_count, read_point_cloud


@dataclass(frozen=True)
class WalkForwardStep:
    """walk-forward 中的一步。

    `source` 是当前 incremental/submap，target 由运行时的当前 world 决定。
    """

    name: str
    source: Path
    output_world: str


@dataclass(frozen=True)
class DatasetSpec:
    """数据集配置展开后的结构。"""

    name: str
    root: Path
    initial_world: Path
    steps: list[WalkForwardStep]


def resolve_path(root: Path, value: str | Path) -> Path:
    """把配置中的相对路径解析到数据集 root 下。"""

    path = Path(value)
    return path if path.is_absolute() else root / path


def load_dataset(config: dict[str, Any]) -> DatasetSpec:
    """从 YAML 字典构造 DatasetSpec。

    使用显式 step 列表，不根据文件名猜顺序，方便未来加入更多序列或跳帧实验。
    """

    root = Path(config.get("root", "."))
    steps = [
        WalkForwardStep(
            name=str(item["name"]),
            source=resolve_path(root, item["source"]),
            output_world=str(item["output_world"]),
        )
        for item in config.get("steps", [])
    ]
    return DatasetSpec(
        name=str(config.get("name", "dataset")),
        root=root,
        initial_world=resolve_path(root, config["initial_world"]),
        steps=steps,
    )


def validate_dataset(dataset: DatasetSpec) -> dict:
    """验证数据集中的 PLY 是否可读，并返回点数摘要。"""

    files = [dataset.initial_world] + [step.source for step in dataset.steps]
    records = []
    for path in files:
        cloud = read_point_cloud(path)
        records.append({"path": str(path), "points": point_count(cloud)})
    return {
        "name": dataset.name,
        "initial_world": str(dataset.initial_world),
        "step_count": len(dataset.steps),
        "files": records,
    }

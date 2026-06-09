from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.pointcloud import point_count, read_point_cloud


@dataclass(frozen=True)
class WalkForwardStep:
    name: str
    source: Path
    output_world: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    initial_world: Path
    steps: list[WalkForwardStep]


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_dataset(config: dict[str, Any]) -> DatasetSpec:
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

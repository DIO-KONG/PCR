from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pcr.domain import BatchRef, RegistrationTask, SequenceTask


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置文件。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_batch_ref(payload: dict[str, Any], *, batch_id_key: str = "id") -> BatchRef:
    """把配置中的 batch 字段转换成 BatchRef。

    baseline 和 step 里字段名略有不同，因此通过 `batch_id_key` 指定 ID 字段。
    """

    return BatchRef(
        batch_id=str(payload[batch_id_key]),
        npz_path=Path(payload["npz"]),
        raw_cloud_path=Path(payload["raw_cloud"]),
        preprocessed_cloud_path=Path(payload["cloud"]),
    )


def build_step_source_ref(step: dict[str, Any]) -> BatchRef:
    """从 step 配置构造 source batch。"""

    return BatchRef(
        batch_id=str(step["source_id"]),
        npz_path=Path(step["source_npz"]),
        raw_cloud_path=Path(step["source_raw_cloud"]),
        preprocessed_cloud_path=Path(step["source_cloud"]),
    )


def build_step_target_ref(step: dict[str, Any]) -> BatchRef:
    """从 step 配置构造 target batch。"""

    return BatchRef(
        batch_id=str(step["target_id"]),
        npz_path=Path(step["target_npz"]),
        raw_cloud_path=Path(step["target_raw_cloud"]),
        preprocessed_cloud_path=Path(step["target_cloud"]),
    )


def build_sequence_task(config: dict[str, Any]) -> SequenceTask:
    """把 walk-forward YAML 转换成显式 SequenceTask。"""

    if "baseline" not in config:
        raise ValueError("Sequence config must contain baseline.")
    if "steps" not in config or not config["steps"]:
        raise ValueError("Sequence config must contain at least one step.")
    if "algorithm" not in config:
        raise ValueError("Sequence config must contain algorithm parameters.")

    baseline = build_batch_ref(config["baseline"])
    tasks: list[RegistrationTask] = []
    for index, step in enumerate(config["steps"], 1):
        tasks.append(
            RegistrationTask(
                step_id=str(step.get("id", f"step_{index:02d}")),
                step_index=index,
                source=build_step_source_ref(step),
                target=build_step_target_ref(step),
                params=config["algorithm"],
            )
        )

    return SequenceTask(
        name=str(config.get("name", "shared_frame_walkforward")),
        baseline=baseline,
        steps=tuple(tasks),
        params=config["algorithm"],
    )


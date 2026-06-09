from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，并要求顶层是 mapping。

    实验框架把 dataset、algorithm、experiment 都写成 YAML。
    统一入口有助于之后加入配置校验或默认值填充。
    """

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置。

    `algorithm_overrides` 和 fusion sweep 都依赖这个函数：只覆盖指定字段，
    未指定的算法参数继续沿用基础配置。
    """

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

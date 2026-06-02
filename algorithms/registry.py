from __future__ import annotations

from typing import Dict

from algorithms.base import AlgorithmSpec
from algorithms import ransac_only


_REGISTRY: Dict[str, AlgorithmSpec] = {}


def register(spec: AlgorithmSpec) -> None:
    if spec.name in _REGISTRY:
        raise ValueError(f"Algorithm already registered: {spec.name}")
    _REGISTRY[spec.name] = spec


def get_algorithm(name: str) -> AlgorithmSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown algorithm '{name}'. Available algorithms: {available}") from exc


def list_algorithms() -> list[str]:
    return sorted(_REGISTRY)


register(
    AlgorithmSpec(
        name=ransac_only.METHOD_NAME,
        runner=ransac_only.run,
        description="Open3D FPFH + RANSAC feature matching baseline.",
        required_dependencies=("open3d", "numpy"),
    )
)

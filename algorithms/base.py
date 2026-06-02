from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np


Status = str
AlgorithmRunner = Callable[[Any, Any, Mapping[str, Any]], "RegistrationResult"]


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    runner: AlgorithmRunner
    description: str = ""
    required_dependencies: tuple[str, ...] = ()


@dataclass
class RegistrationResult:
    method: str
    status: Status
    transformation: Optional[np.ndarray]
    runtime_sec: float = 0.0
    fitness: Optional[float] = None
    inlier_rmse: Optional[float] = None
    correspondence_set_size: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    algorithm_metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status in {"failed", "skipped"}:
            self.transformation = None
            return
        if self.transformation is None:
            raise ValueError("Successful RegistrationResult must include a 4x4 transformation.")
        self.transformation = np.asarray(self.transformation, dtype=float)
        if self.transformation.shape != (4, 4):
            raise ValueError("RegistrationResult.transformation must be a 4x4 matrix.")

    @property
    def has_valid_transform(self) -> bool:
        return self.transformation is not None and self.status == "success"

    @classmethod
    def skipped(cls, method: str, reason: str, params: Optional[Mapping[str, Any]] = None) -> "RegistrationResult":
        return cls(
            method=method,
            status="skipped",
            transformation=None,
            params=dict(params or {}),
            error=reason,
        )

    @classmethod
    def failed(cls, method: str, error: str, params: Optional[Mapping[str, Any]] = None) -> "RegistrationResult":
        return cls(
            method=method,
            status="failed",
            transformation=None,
            params=dict(params or {}),
            error=error,
        )

    def to_record(self) -> Dict[str, Any]:
        algorithm_metrics = dict(self.algorithm_metrics)
        if self.fitness is not None:
            algorithm_metrics.setdefault("fitness", self.fitness)
        if self.inlier_rmse is not None:
            algorithm_metrics.setdefault("inlier_rmse", self.inlier_rmse)
        if self.correspondence_set_size is not None:
            algorithm_metrics.setdefault("correspondence_set_size", self.correspondence_set_size)

        record: Dict[str, Any] = {
            "method": self.method,
            "status": self.status,
            "has_valid_transform": self.has_valid_transform,
            "algorithm_time": self.runtime_sec,
            "params": self.params,
            "error": self.error,
            "metadata": self.metadata,
        }
        for key, value in algorithm_metrics.items():
            record[f"algorithm_{key}"] = value
        return record

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
    transformation: np.ndarray
    runtime_sec: float = 0.0
    fitness: Optional[float] = None
    inlier_rmse: Optional[float] = None
    correspondence_set_size: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.transformation = np.asarray(self.transformation, dtype=float)
        if self.transformation.shape != (4, 4):
            raise ValueError("RegistrationResult.transformation must be a 4x4 matrix.")

    @classmethod
    def skipped(cls, method: str, reason: str, params: Optional[Mapping[str, Any]] = None) -> "RegistrationResult":
        return cls(
            method=method,
            status="skipped",
            transformation=np.eye(4),
            params=dict(params or {}),
            error=reason,
        )

    @classmethod
    def failed(cls, method: str, error: str, params: Optional[Mapping[str, Any]] = None) -> "RegistrationResult":
        return cls(
            method=method,
            status="failed",
            transformation=np.eye(4),
            params=dict(params or {}),
            error=error,
        )

    def to_record(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "status": self.status,
            "runtime_sec": self.runtime_sec,
            "fitness": self.fitness,
            "inlier_rmse": self.inlier_rmse,
            "correspondence_set_size": self.correspondence_set_size,
            "params": self.params,
            "error": self.error,
            "metadata": self.metadata,
        }

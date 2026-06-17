from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RegistrationCandidate:
    """一个配准候选。"""

    candidate_id: int
    matrix: np.ndarray
    score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationResult:
    """一个算法方案的统一输出。"""

    algorithm: str
    source_path: Path
    target_path: Path
    status: str
    matrix: np.ndarray
    metrics: dict[str, Any]
    candidates: list[RegistrationCandidate] = field(default_factory=list)
    message: str = ""

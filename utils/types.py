from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Candidate:
    candidate_id: int
    matrix: np.ndarray
    coarse_score: float = 0.0
    cluster_id: int | None = None
    cluster_size: int | None = None
    cluster_top_score_count: int | None = None
    scale_values: dict[str, float] = field(default_factory=dict)
    vote_peak_ratio: float = 0.0
    metrics: dict[str, float | None] = field(default_factory=dict)
    rank_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationResult:
    algorithm: str
    source_path: Path
    target_path: Path
    matrix: np.ndarray
    metrics: dict[str, float | str | None]
    status: str
    top_candidates: list[Candidate] = field(default_factory=list)
    debug_artifacts: dict[str, str] = field(default_factory=dict)

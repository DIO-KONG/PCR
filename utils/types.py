from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Candidate:
    """单个配准候选。

    topview_vote 会生成很多候选，每个候选记录自己的矩阵、粗分数、精修后的尺度、
    投票质量和评估指标。后续保存 `top_candidates.md` 时也依赖这个结构。
    """

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
    """一次 pairwise 注册的统一返回值。

    所有算法都应该返回这个结构，testbench 因此不需要关心算法内部是 RANSAC、
    top-view voting，还是未来新增的 surfel / pose graph 初始化方法。
    """

    algorithm: str
    source_path: Path
    target_path: Path
    matrix: np.ndarray
    metrics: dict[str, float | str | None]
    status: str
    top_candidates: list[Candidate] = field(default_factory=list)
    debug_artifacts: dict[str, str] = field(default_factory=dict)

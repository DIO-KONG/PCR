from __future__ import annotations

from typing import Any


def score_metrics(metrics: dict[str, Any]) -> float:
    """把严格评估指标压成单个候选分数。

    分数只用于候选排序，不代表几何真值。这里故意惩罚 median/p90 误差，
    避免仅靠较宽松 inlier 数量把局部错误解排到最前。
    """

    return float(metrics["inlier_ratio"] - metrics["median_error"] - 0.25 * metrics["p90_error"])


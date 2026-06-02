from __future__ import annotations

from typing import Any, Iterable, Mapping


def score_record(record: Mapping[str, Any]) -> float:
    status = record.get("status")
    if status != "success":
        return float("-inf")
    fitness = float(record.get("eval_fitness") or record.get("algorithm_fitness") or 0.0)
    rmse = float(record.get("eval_inlier_rmse") or record.get("algorithm_inlier_rmse") or 0.0)
    return fitness - rmse


def select_top_k(records: Iterable[Mapping[str, Any]], top_k: int) -> list[dict[str, Any]]:
    ranked = sorted((dict(record) for record in records), key=score_record, reverse=True)
    return ranked[: max(0, int(top_k))]


def mark_free(record: Mapping[str, Any], threshold: Mapping[str, Any] | None = None) -> dict[str, Any]:
    threshold = dict(threshold or {})
    min_fitness = float(threshold.get("min_fitness", 0.0))
    max_rmse = float(threshold.get("max_rmse", float("inf")))
    marked = dict(record)
    fitness = float(marked.get("eval_fitness") or marked.get("algorithm_fitness") or 0.0)
    rmse = float(marked.get("eval_inlier_rmse") or marked.get("algorithm_inlier_rmse") or float("inf"))
    marked["is_free"] = marked.get("status") != "success" or fitness < min_fitness or rmse > max_rmse
    return marked

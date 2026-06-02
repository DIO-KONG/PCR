from __future__ import annotations

from typing import Any, Mapping

from workflows.assignment import mark_free, select_top_k
from workflows.pairwise import run_pairwise_task


def run_multi_to_multi_task(task: Mapping[str, Any], default_config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    all_records: list[dict[str, Any]] = []
    assignment = dict(task.get("assignment", {}))
    top_k = int(assignment.get("top_k", 1))
    threshold = assignment.get("free_threshold", {})

    for source in task.get("sources", []):
        candidates = []
        for target in task.get("targets", []):
            pair_task = dict(task)
            pair_task["source"] = source
            pair_task["target"] = target
            pair_task["task"] = {
                "id": f"{task.get('task', {}).get('id', 'multi_to_multi')}__{source.get('id', 'source')}__{target.get('id', 'target')}",
                "type": "pairwise",
            }
            candidates.append(run_pairwise_task(pair_task, default_config))
        all_records.extend(mark_free(record, threshold) for record in select_top_k(candidates, top_k))
    return all_records

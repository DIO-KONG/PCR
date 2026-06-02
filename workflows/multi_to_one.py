from __future__ import annotations

from typing import Any, Mapping

from workflows.assignment import mark_free, select_top_k
from workflows.pairwise import run_pairwise_task


def run_multi_to_one_task(task: Mapping[str, Any], default_config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    records = []
    target = task["target"]
    for source in task.get("sources", []):
        pair_task = dict(task)
        pair_task["source"] = source
        pair_task["target"] = target
        pair_task["task"] = {"id": f"{task.get('task', {}).get('id', 'multi_to_one')}__{source.get('id', 'source')}", "type": "pairwise"}
        records.append(run_pairwise_task(pair_task, default_config))

    assignment = dict(task.get("assignment", {}))
    top_k = int(assignment.get("top_k", len(records)))
    threshold = assignment.get("free_threshold", {})
    return [mark_free(record, threshold) for record in select_top_k(records, top_k)]

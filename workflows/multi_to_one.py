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
    threshold = assignment.get("free_threshold", {})
    marked = [mark_free(record, threshold) for record in records]
    if assignment.get("mode") == "top_k" or assignment.get("apply_top_k", False):
        top_k = int(assignment.get("top_k", len(marked)))
        return select_top_k(marked, top_k)
    return marked

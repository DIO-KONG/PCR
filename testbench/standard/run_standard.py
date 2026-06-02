from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from algorithms.registry import list_algorithms
from utils import io, reporting
from workflows.pairwise import run_pairwise_task


def run_standard(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    task_path = Path("data/tasks/pair_mission1.yaml")
    task = io.read_config(task_path)
    run_dir = reporting.make_run_dir(Path("results") / "standard", "standard")
    records = []
    for algorithm_name in list_algorithms():
        task_for_algorithm = dict(task)
        task_for_algorithm["algorithm"] = {
            "name": algorithm_name,
            "params": dict(config.get("algorithm", {}).get("params", {})),
        }
        task_for_algorithm["output"] = {"category": "standard"}
        records.append(run_pairwise_task(task_for_algorithm, config, output_dir=run_dir, write_report=False))
    reporting.write_metrics(run_dir, records)
    reporting.write_markdown_report(run_dir / "report" / "summary.md", records)
    return records

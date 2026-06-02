from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils import io, reporting
from workflows.pairwise import run_pairwise_task


EXPERIMENT_NAME = "ablation"

VARIANTS = [
    ("ransac_only", "ransac_only", {}),
    ("ransac_best_of_5_only", "gicp_refine", {"ransac_trials": 5, "enable_refine": False}),
    ("single_ransac_gicp", "gicp_refine", {"ransac_trials": 1, "enable_refine": True}),
    ("best_of_5_ransac_gicp", "gicp_refine", {"ransac_trials": 5, "enable_refine": True}),
]


def run_experiment(config: Mapping[str, Any], task: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    run_dir = reporting.make_run_dir(Path("results") / "specified" / "gicp_refine" / EXPERIMENT_NAME, EXPERIMENT_NAME)
    records: list[dict[str, Any]] = []
    for variant_name, algorithm_name, params in VARIANTS:
        records.append(_run_variant(config, task, run_dir, variant_name, algorithm_name, params))
    _write_outputs(run_dir, records)
    return run_dir, records


def _run_variant(
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    run_dir: Path,
    variant_name: str,
    algorithm_name: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    pair_task = _task_with_algorithm(task, algorithm_name, params)
    pair_task["source"] = _source_with_variant(pair_task["source"], variant_name)
    pair_task["task"] = {"id": f"{EXPERIMENT_NAME}__{variant_name}", "type": "pairwise"}
    record = run_pairwise_task(pair_task, config, output_dir=run_dir, write_report=False)
    record.update({"experiment_name": EXPERIMENT_NAME, "variant_name": variant_name})
    return record


def _task_with_algorithm(task: Mapping[str, Any], algorithm_name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    pair_task = dict(task)
    base_algorithm = dict(pair_task.get("algorithm", {}))
    base_params = dict(base_algorithm.get("params", {}))
    base_params.update(params)
    pair_task["algorithm"] = {"name": algorithm_name, "params": base_params}
    pair_task["output"] = {"category": "specified", "experiment": EXPERIMENT_NAME}
    return pair_task


def _source_with_variant(source: Mapping[str, Any], suffix: str) -> dict[str, Any]:
    updated = dict(source)
    updated["id"] = f"{source.get('id', 'source')}__{suffix}"
    return updated


def _write_outputs(run_dir: Path, records: list[dict[str, Any]]) -> None:
    reporting.write_metrics(run_dir, records)
    reporting.write_markdown_report(run_dir / "summary.md", records)
    reporting.write_markdown_report(run_dir / "report" / "summary.md", records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run gicp_refine ablation experiment.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", default="data/tasks/pair_mission1.yaml")
    args = parser.parse_args()
    run_dir, records = run_experiment(io.read_config(args.config), io.read_config(args.task))
    print(f"wrote {len(records)} records to {run_dir}")
    return 0 if all(record.get("status") == "success" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

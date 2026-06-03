from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils import io, reporting
from workflows.pairwise import run_pairwise_task


EXPERIMENT_NAME = "pair_batch_parameter_test"

PARAMETER_SETS = [
    ("rigid_none", {"coarse_method": "ransac", "coarse_trials": 3, "affine_mode": "none"}),
    ("constrained_affine", {"coarse_method": "ransac", "coarse_trials": 3, "affine_mode": "constrained"}),
    ("unconstrained_affine", {"coarse_method": "ransac", "coarse_trials": 3, "affine_mode": "unconstrained"}),
]


def run_experiment(config: Mapping[str, Any], task: Mapping[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    _validate_pair_paths(task)
    run_dir = reporting.make_run_dir(Path("results") / "specified" / "multi_comb" / EXPERIMENT_NAME, EXPERIMENT_NAME)
    records: list[dict[str, Any]] = []
    for pair in task.get("pairs", []):
        for variant_name, params in PARAMETER_SETS:
            records.append(_run_variant(config, task, pair, run_dir, variant_name, params))
    reporting.write_metrics(run_dir, records)
    _write_summary(run_dir / "summary.md", records)
    _write_summary(run_dir / "report" / "summary.md", records)
    _write_visualization_commands(run_dir / "report" / "visualization_commands.md", records)
    return run_dir, records


def _validate_pair_paths(task: Mapping[str, Any]) -> None:
    missing = []
    for pair in task.get("pairs", []):
        for role in ("source", "target"):
            path = Path(pair[role]["path"])
            if not path.exists():
                missing.append(f"{pair.get('id')} {role}: {path}")
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(f"Missing point cloud path(s):\n{joined}")


def _run_variant(
    config: Mapping[str, Any],
    task: Mapping[str, Any],
    pair: Mapping[str, Any],
    run_dir: Path,
    variant_name: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    pair_task = {
        "task": {"id": f"{pair.get('id')}__{variant_name}", "type": "pairwise"},
        "source": _source_with_variant(pair["source"], variant_name),
        "target": pair["target"],
        "algorithm": {"name": "multi_comb", "params": _merged_algorithm_params(config, task, params)},
        "output": {"category": "specified", "experiment": EXPERIMENT_NAME},
    }
    record = run_pairwise_task(pair_task, config, output_dir=run_dir, write_report=False)
    record.update({"experiment_name": EXPERIMENT_NAME, "pair_id": pair.get("id"), "variant_name": variant_name})
    return record


def _source_with_variant(source: Mapping[str, Any], suffix: str) -> dict[str, Any]:
    updated = dict(source)
    updated["id"] = f"{source.get('id', 'source')}__{suffix}"
    return updated


def _merged_algorithm_params(config: Mapping[str, Any], task: Mapping[str, Any], params: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(config.get("algorithm", {}).get("params", {}))
    merged.update(dict(task.get("algorithm", {}).get("params", {})))
    merged.update(dict(params))
    return merged


def _write_summary(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# multi_comb Pair Batch Parameter Test", ""]
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_pair.setdefault(str(record.get("pair_id")), []).append(record)
    for pair_id, pair_records in by_pair.items():
        lines.extend([f"## {pair_id}", ""])
        for title, record in [
            ("Best by eval_fitness", _best(pair_records, "eval_fitness", reverse=True)),
            ("Best by eval_trimmed_mean_nn_dist", _best(pair_records, "eval_trimmed_mean_nn_dist", reverse=False)),
            ("Best by weighted score", _best(pair_records, "algorithm_best_score", reverse=True)),
        ]:
            lines.extend(_summary_block(title, record))
        lines.extend(["| variant | status | transform_type | affine | scale_values | eval_fitness | eval_trimmed_mean_nn_dist | weighted_score | matrix |", "|---|---|---|---|---|---:|---:|---:|---|"])
        for record in pair_records:
            lines.append(
                "| {variant} | {status} | {transform_type} | {affine} | {scale} | {fitness} | {trimmed} | {score} | {matrix} |".format(
                    variant=record.get("variant_name"),
                    status=record.get("status"),
                    transform_type=record.get("algorithm_transform_type"),
                    affine=record.get("algorithm_affine_mode") != "none",
                    scale=record.get("algorithm_scale_values"),
                    fitness=_fmt(record.get("eval_fitness")),
                    trimmed=_fmt(record.get("eval_trimmed_mean_nn_dist")),
                    score=_fmt(record.get("algorithm_best_score")),
                    matrix=record.get("matrix_path"),
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_block(title: str, record: Mapping[str, Any] | None) -> list[str]:
    if record is None:
        return [f"### {title}", "", "- No valid result.", ""]
    return [
        f"### {title}",
        "",
        f"- variant: `{record.get('variant_name')}`",
        f"- status: `{record.get('status')}`",
        f"- transform_type: `{record.get('algorithm_transform_type')}`",
        f"- affine: `{record.get('algorithm_affine_mode') != 'none'}`",
        f"- scale_values: `{record.get('algorithm_scale_values')}`",
        f"- eval_fitness: `{_fmt(record.get('eval_fitness'))}`",
        f"- eval_trimmed_mean_nn_dist: `{_fmt(record.get('eval_trimmed_mean_nn_dist'))}`",
        f"- weighted_score: `{_fmt(record.get('algorithm_best_score'))}`",
        "",
    ]


def _best(records: list[dict[str, Any]], key: str, reverse: bool) -> dict[str, Any] | None:
    valid = [record for record in records if record.get("status") == "success" and record.get(key) is not None]
    if not valid:
        return None
    return sorted(valid, key=lambda record: float(record[key]), reverse=reverse)[0]


def _write_visualization_commands(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Visualization Commands", ""]
    skipped: list[str] = []
    for record in records:
        reason = _visualization_skip_reason(record)
        if reason:
            skipped.append(f"- {record.get('pair_id')} / {record.get('variant_name')}: {reason}")
            continue
        lines.extend(
            [
                f"## {record.get('pair_id')} / {record.get('variant_name')}",
                "",
                f"- reason: eval_fitness={_fmt(record.get('eval_fitness'))}, trimmed={_fmt(record.get('eval_trimmed_mean_nn_dist'))}, transform_type={record.get('algorithm_transform_type')}",
                f"- matrix: `{record.get('matrix_path')}`",
                f"- cloud: `{record.get('cloud_path')}`",
                "",
                "```bat",
                ".env\\python.exe scripts\\visualize_overlay.py --source \"{source}\" --target \"{target}\" --matrix \"{matrix}\"".format(
                    source=record.get("source_path"),
                    target=record.get("target_path"),
                    matrix=record.get("matrix_path"),
                ),
                "```",
                "",
            ]
        )
    if skipped:
        lines.extend(["## Skipped Low-Quality Results", "", *skipped, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _visualization_skip_reason(record: Mapping[str, Any]) -> str | None:
    if record.get("status") != "success" or not record.get("matrix_path"):
        return "no successful matrix artifact"
    fitness = record.get("eval_fitness")
    trimmed = record.get("eval_trimmed_mean_nn_dist")
    if fitness is not None and float(fitness) < 0.5:
        return f"eval_fitness too low ({_fmt(fitness)})"
    if trimmed is not None and float(trimmed) > 1.0:
        return f"eval_trimmed_mean_nn_dist too high ({_fmt(trimmed)})"
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multi_comb batch pair parameter test.")
    parser.add_argument("--task", default="data/tasks/pair_batch_incremental_world.yaml")
    parser.add_argument("--config", default="configs/multi_comb.yaml")
    args = parser.parse_args()
    run_dir, records = run_experiment(io.read_config(args.config), io.read_config(args.task))
    print(f"wrote {len(records)} records to {run_dir}")
    return 0 if all(record.get("status") == "success" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

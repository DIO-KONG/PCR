from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def make_run_dir(root: str | Path, prefix: str = "run") -> Path:
    run_dir = Path(root) / make_run_id(prefix)
    for child in ("cloud", "matrix", "report"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    return run_dir


def combine_record(task: Mapping[str, Any], result: Any, metrics: Mapping[str, Any], artifacts: Mapping[str, Any]) -> dict[str, Any]:
    source = task.get("source", {})
    target = task.get("target", {})
    record = {
        "task_id": task.get("task", {}).get("id"),
        "source_id": source.get("id"),
        "target_id": target.get("id"),
        "source_path": source.get("path"),
        "target_path": target.get("path"),
    }
    record.update(result.to_record())
    record.update(metrics)
    record.update(artifacts)
    return record


def write_metrics(output_dir: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    output_dir = Path(output_dir)
    rows = [dict(record) for record in records]
    _write_json(output_dir / "metrics.json", rows)
    _write_csv(output_dir / "metrics.csv", rows)


def write_markdown_report(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(record) for record in records]
    lines = ["# PCR Report", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row.get('task_id') or 'task'}",
                "",
                f"- Algorithm: `{row.get('method')}`",
                f"- Status: `{row.get('status')}`",
                f"- Source: `{row.get('source_id')}`",
                f"- Target: `{row.get('target_id')}`",
                f"- Fitness: `{row.get('fitness')}`",
                f"- RMSE: `{row.get('inlier_rmse') or row.get('rmse')}`",
                f"- Matrix: `{row.get('matrix_path')}`",
                f"- Cloud: `{row.get('cloud_path')}`",
                f"- Error: `{row.get('error')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value

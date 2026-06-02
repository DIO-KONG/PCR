from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from algorithms.base import RegistrationResult
from algorithms.registry import get_algorithm
from utils import io, metrics, preprocessing, reporting, visualization


def run_pairwise_task(task: Mapping[str, Any], default_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = dict(default_config or {})
    algorithm_config = dict(config.get("algorithm", {}))
    algorithm_config.update(dict(task.get("algorithm", {})))
    algorithm_name = algorithm_config.get("name", "ransac_only")
    algorithm_params = dict(algorithm_config.get("params", {}))

    source = task["source"]
    target = task["target"]
    prep_config = dict(config.get("preprocessing", {}))
    if "voxel_size" not in algorithm_params and "voxel_size" in prep_config:
        algorithm_params["voxel_size"] = prep_config["voxel_size"]

    spec = get_algorithm(algorithm_name)
    source_pcd = None
    target_pcd = None
    metric_record = {}
    try:
        source_pcd = io.read_point_cloud(source["path"])
        target_pcd = io.read_point_cloud(target["path"])
        prepared_source = preprocessing.prepare_point_cloud(source_pcd, prep_config)
        prepared_target = preprocessing.prepare_point_cloud(target_pcd, prep_config)
        result = spec.runner(prepared_source, prepared_target, algorithm_params)
        metric_record = metrics.evaluate_registration(
            source_pcd,
            target_pcd,
            result.transformation,
            dict(config.get("metrics", {})),
        )
    except RuntimeError as exc:
        message = str(exc)
        if "open3d" in message.lower():
            result = RegistrationResult.skipped(algorithm_name, message, algorithm_params)
        else:
            result = RegistrationResult.failed(algorithm_name, message, algorithm_params)
    except Exception as exc:
        result = RegistrationResult.failed(algorithm_name, str(exc), algorithm_params)

    output_dir = reporting.make_run_dir(
        _resolve_output_root(task, algorithm_name),
        task.get("task", {}).get("id", "pairwise"),
    )
    artifacts = _write_artifacts(output_dir, source_pcd, target_pcd, result, metric_record, source, target)
    record = reporting.combine_record(task, result, metric_record, artifacts)
    reporting.write_metrics(output_dir, [record])
    reporting.write_markdown_report(output_dir / "report" / "summary.md", [record])
    return record


def _resolve_output_root(task: Mapping[str, Any], algorithm_name: str) -> Path:
    output = dict(task.get("output", {}))
    category = output.get("category", "specified")
    if category == "standard":
        return Path("results") / "standard"
    experiment = output.get("experiment", "manual")
    return Path("results") / "specified" / algorithm_name / experiment


def _write_artifacts(
    output_dir: Path,
    source_pcd: Any,
    target_pcd: Any,
    result: Any,
    metric_record: Mapping[str, Any],
    source_meta: Mapping[str, Any],
    target_meta: Mapping[str, Any],
) -> dict[str, str]:
    matrix_path = output_dir / "matrix" / f"{source_meta.get('id', 'source')}__to__{target_meta.get('id', 'target')}.txt"
    io.write_matrix(matrix_path, result.transformation)

    artifacts = {"matrix_path": str(matrix_path)}
    if result.status == "success" and source_pcd is not None and target_pcd is not None:
        transformed_path = output_dir / "cloud" / f"{source_meta.get('id', 'source')}__registered.ply"
        overlay_path = output_dir / "cloud" / f"{source_meta.get('id', 'source')}__overlay.ply"
        visualization.save_transformed_cloud(source_pcd, result.transformation, transformed_path)
        visualization.save_overlay_cloud(source_pcd, target_pcd, result.transformation, overlay_path)
        artifacts.update({"cloud_path": str(transformed_path), "overlay_path": str(overlay_path)})
    artifacts["orthogonality_error"] = str(metric_record.get("orthogonality_error"))
    return artifacts

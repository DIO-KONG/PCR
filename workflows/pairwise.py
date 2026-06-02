from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Mapping

from algorithms.base import RegistrationResult
from algorithms.registry import get_algorithm
from utils import io, metrics, preprocessing, reporting, visualization


def run_pairwise_task(
    task: Mapping[str, Any],
    default_config: Mapping[str, Any] | None = None,
    output_dir: str | Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    timings = {
        "preprocess_time": 0.0,
        "algorithm_time": 0.0,
        "evaluation_time": 0.0,
        "artifact_time": 0.0,
        "total_time": 0.0,
    }
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
    metric_record = metrics.empty_registration_metrics()
    try:
        preprocess_started = time.perf_counter()
        prepared_source = preprocessing.prepare_point_cloud_from_path_with_cache(source["path"], prep_config)
        prepared_target = preprocessing.prepare_point_cloud_from_path_with_cache(target["path"], prep_config)
        source_pcd = prepared_source["raw_pcd"]
        target_pcd = prepared_target["raw_pcd"]
        timings["preprocess_time"] = time.perf_counter() - preprocess_started

        algorithm_started = time.perf_counter()
        result = spec.runner(prepared_source, prepared_target, algorithm_params)
        timings["algorithm_time"] = result.runtime_sec or (time.perf_counter() - algorithm_started)
        if result.status == "success" and result.has_valid_transform:
            evaluation_started = time.perf_counter()
            metric_record = metrics.evaluate_registration(
                source_pcd,
                target_pcd,
                result.transformation,
                dict(config.get("metrics", {})),
                registration_result=result,
            )
            timings["evaluation_time"] = time.perf_counter() - evaluation_started
        else:
            metric_record = metrics.empty_registration_metrics(result.error)
    except RuntimeError as exc:
        message = str(exc)
        if "open3d" in message.lower():
            result = RegistrationResult.skipped(algorithm_name, message, algorithm_params)
        else:
            result = RegistrationResult.failed(algorithm_name, message, algorithm_params)
        metric_record = metrics.empty_registration_metrics(message)
    except Exception as exc:
        result = RegistrationResult.failed(algorithm_name, str(exc), algorithm_params)
        metric_record = metrics.empty_registration_metrics(str(exc))

    if output_dir is None:
        output_dir = reporting.make_run_dir(
            _resolve_output_root(task, algorithm_name),
            task.get("task", {}).get("id", "pairwise"),
        )
    else:
        output_dir = reporting.ensure_run_dirs(output_dir)
    artifact_started = time.perf_counter()
    artifacts = _write_artifacts(output_dir, source_pcd, target_pcd, result, metric_record, source, target)
    timings["artifact_time"] = time.perf_counter() - artifact_started
    timings["total_time"] = time.perf_counter() - total_started
    record = reporting.combine_record(task, result, metric_record, artifacts)
    record.update(timings)
    if write_report:
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
    if result.has_valid_transform and source_pcd is not None and target_pcd is not None:
        stem = f"{result.method}__{source_meta.get('id', 'source')}__to__{target_meta.get('id', 'target')}"
        matrix_path = output_dir / "matrix" / f"{stem}.txt"
        transformed_path = output_dir / "cloud" / f"{stem}__registered.ply"
        overlay_path = output_dir / "cloud" / f"{stem}__overlay.ply"
        io.write_matrix(matrix_path, result.transformation)
        visualization.save_transformed_cloud(source_pcd, result.transformation, transformed_path)
        visualization.save_overlay_cloud(source_pcd, target_pcd, result.transformation, overlay_path)
        return {
            "matrix_path": str(matrix_path),
            "cloud_path": str(transformed_path),
            "overlay_path": str(overlay_path),
        }
    return {}

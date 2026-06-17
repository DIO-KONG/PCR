#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.registry import get_algorithm
from utils.pointcloud import load_cloud, make_overlay, save_cloud, transform_cloud


DEFAULT_EXPERIMENT = Path("testbench/configs/experiments/registration_schemes.yaml")


def json_safe(value: Any) -> Any:
    """转换 numpy/dataclass/path，便于写 JSON。"""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    return value


def load_yaml(path: str | Path) -> dict:
    """读取 YAML 配置。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: str | Path, payload: dict) -> None:
    """写 JSON。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_matrix(path: str | Path, matrix: np.ndarray) -> None:
    """写 4x4 矩阵。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.10f")


def make_run_dir(result_root: Path, run_name: str | None, overwrite: bool) -> Path:
    """创建实验输出目录。"""

    if run_name is None:
        run_name = datetime.now().strftime("registration_%Y%m%d_%H%M%S")
    run_dir = result_root / run_name
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}. Use --overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_candidate_table(path: Path, result) -> None:
    """保存 top candidates Markdown 表。"""

    if result.algorithm == "shared_frame_alignment":
        lines = [
            "| rank | candidate | coarse_threshold | score | status | inlier_ratio | inliers | median_error | p90_error | icp_status | method |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
        ]
        for rank, candidate in enumerate(result.candidates, 1):
            metrics = candidate.metrics
            icp = metrics.get("icp_refinement") or {}
            lines.append(
                "| {rank} | {cid} | {coarse:.3f} | {score:.6f} | {status} | {ratio:.6f} | {inliers} | {median:.6f} | {p90:.6f} | {icp_status} | {method} |".format(
                    rank=rank,
                    cid=candidate.candidate_id,
                    coarse=float(metrics.get("coarse_threshold") or 0.0),
                    score=float(candidate.score),
                    status=metrics.get("status", ""),
                    ratio=float(metrics.get("inlier_ratio") or 0.0),
                    inliers=int(metrics.get("inlier_count") or 0),
                    median=float(metrics.get("median_error") or 999.0),
                    p90=float(metrics.get("p90_error") or 999.0),
                    icp_status=icp.get("status", ""),
                    method=candidate.metadata.get("method", ""),
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines = [
        "| rank | candidate | score | status | src_fit | bidir_fit | trimmed | method |",
        "|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for rank, candidate in enumerate(result.candidates, 1):
        metrics = candidate.metrics
        lines.append(
            "| {rank} | {cid} | {score:.6f} | {status} | {src:.6f} | {bidir:.6f} | {trimmed:.6f} | {method} |".format(
                rank=rank,
                cid=candidate.candidate_id,
                score=float(candidate.score),
                status=metrics.get("status", ""),
                src=float(metrics.get("source_fitness") or 0.0),
                bidir=float(metrics.get("bidirectional_fitness_hmean") or 0.0),
                trimmed=float(metrics.get("source_trimmed_mean_nn") or 999.0),
                method=candidate.metadata.get("coarse_method", candidate.metadata.get("refine_method", "")),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_scheme_outputs(output_dir: Path, result, source_path: Path, target_path: Path, config: dict) -> dict:
    """保存单个算法方案输出。"""

    source_cloud_path = Path(config.get("source_cloud", source_path))
    target_cloud_path = Path(config.get("target_cloud", target_path))
    source = load_cloud(source_cloud_path)
    target = load_cloud(target_cloud_path)
    registered = transform_cloud(source, result.matrix)
    overlay = make_overlay(target, registered)

    write_matrix(output_dir / "matrix.txt", result.matrix)
    write_json(
        output_dir / "metrics.json",
        {
            "algorithm": result.algorithm,
            "status": result.status,
            "message": result.message,
            "source": source_path,
            "target": target_path,
            "source_cloud": source_cloud_path,
            "target_cloud": target_cloud_path,
            "metrics": result.metrics,
            "candidates": result.candidates,
        },
    )
    save_candidate_table(output_dir / "top_candidates.md", result)
    save_cloud(output_dir / "registered_source.ply", registered)
    save_cloud(output_dir / "overlay.ply", overlay)
    return {
        "algorithm": result.algorithm,
        "status": result.status,
        "message": result.message,
        "metrics": result.metrics,
        "output_dir": str(output_dir),
    }


def run(args: argparse.Namespace) -> dict:
    """运行多个配准方案。"""

    experiment = load_yaml(args.experiment)
    dataset = load_yaml(experiment["dataset_config"])
    source_path = Path(args.source or dataset["source"])
    target_path = Path(args.target or dataset["target"])
    result_root = Path(args.result_root or experiment.get("result_root", "result/registration_schemes"))
    run_dir = make_run_dir(result_root, args.run_name, args.overwrite)

    write_json(
        run_dir / "run_config.json",
        {
            "experiment": experiment,
            "dataset": dataset,
            "source": source_path,
            "target": target_path,
        },
    )

    summaries = []
    for scheme_path in experiment["schemes"]:
        config = load_yaml(scheme_path)
        if args.scheme and config["name"] != args.scheme:
            continue
        algorithm = get_algorithm(config["name"])
        scheme_dir = run_dir / config["name"]
        try:
            result = algorithm(source_path, target_path, config)
            summary = save_scheme_outputs(scheme_dir, result, source_path, target_path, config)
        except Exception as exc:  # noqa: BLE001 - 实验入口要保留失败方案报告。
            summary = {
                "algorithm": config["name"],
                "status": "failed",
                "message": str(exc),
                "output_dir": str(scheme_dir),
            }
            write_json(scheme_dir / "metrics.json", summary)
        summaries.append(summary)
        print(f"[{summary['algorithm']}] {summary['status']} -> {summary['output_dir']}")
        if summary.get("message"):
            print(f"  message: {summary['message']}")

    summary = {
        "run_dir": str(run_dir),
        "source": str(source_path),
        "target": str(target_path),
        "schemes": summaries,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run multiple point-cloud registration schemes.")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--scheme", default=None, help="Run one scheme by config name.")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps(json_safe(result), indent=2, ensure_ascii=False))

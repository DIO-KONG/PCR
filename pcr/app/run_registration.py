from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pcr.algorithms.shared_frame.pairwise import register_shared_frame_pairwise
from pcr.artifacts.store import json_safe
from pcr.config.loader import load_yaml
from pcr.io.pointcloud_io import make_registration_overlay, load_point_cloud, save_point_cloud


DEFAULT_EXPERIMENT = Path("testbench/configs/experiments/registration_schemes.yaml")


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_matrix(path: str | Path, matrix: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.10f")


def make_run_dir(result_root: Path, run_name: str | None, overwrite: bool) -> Path:
    """创建 pairwise 实验输出目录。"""

    if run_name is None:
        run_name = datetime.now().strftime("registration_%Y%m%d_%H%M%S")
    run_dir = result_root / run_name
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}. Use --overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_candidate_table(path: Path, result) -> None:
    """保存 shared-frame top candidates 表。"""

    lines = [
        "| rank | candidate | coarse_threshold | score | status | inlier_ratio | inliers | median_error | p90_error | icp_status | method |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for rank, candidate in enumerate(result.candidates, 1):
        metrics = candidate.metrics.to_dict()
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


def save_scheme_outputs(output_dir: Path, result, source_path: Path, target_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    """保存一个 pairwise 方案的输出 artifact。"""

    source_cloud_path = Path(config.get("source_cloud", source_path))
    target_cloud_path = Path(config.get("target_cloud", target_path))
    source = load_point_cloud(source_cloud_path)
    target = load_point_cloud(target_cloud_path)
    registered = result.transform.apply_cloud(source)
    overlay = make_registration_overlay(target, registered)

    write_matrix(output_dir / "matrix.txt", result.transform.matrix)
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
    save_point_cloud(output_dir / "registered_source.ply", registered)
    save_point_cloud(output_dir / "overlay.ply", overlay)
    return {
        "algorithm": result.algorithm,
        "status": result.status,
        "message": result.message,
        "metrics": result.metrics,
        "output_dir": str(output_dir),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """运行 pairwise registration 方案。"""

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
        scheme_dir = run_dir / config["name"]
        try:
            if config["name"] != "shared_frame_alignment":
                raise ValueError(f"Unknown registration scheme: {config['name']}")
            result = register_shared_frame_pairwise(source_path, target_path, config)
            summary = save_scheme_outputs(scheme_dir, result, source_path, target_path, config)
        except Exception as exc:  # noqa: BLE001 - 实验入口需要报告失败方案。
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple point-cloud registration schemes.")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--scheme", default=None, help="Run one scheme by config name.")
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(json_safe(run(parse_args(argv))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


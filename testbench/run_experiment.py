#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.registry import get_algorithm
from utils.config import deep_merge, load_yaml
from utils.data import load_dataset, validate_dataset
from utils.fusion import fuse_world
from utils.io import save_pair_outputs, write_json, write_point_cloud, write_yaml
from utils.pointcloud import read_point_cloud


DEFAULT_EXPERIMENT = Path("testbench/configs/experiments/basic_walk_forward.yaml")


def load_experiment(path: Path) -> dict:
    experiment = load_yaml(path)
    dataset = load_yaml(experiment["dataset_config"])
    algorithm_config = load_yaml(experiment["algorithm_config"])
    algorithm_config = deep_merge(algorithm_config, experiment.get("algorithm_overrides", {}))
    return {
        "experiment_path": str(path),
        "experiment": experiment,
        "dataset": dataset,
        "algorithm_config": algorithm_config,
    }


def make_run_dir(result_root: Path, run_name: str | None, overwrite: bool) -> Path:
    if run_name is None:
        run_name = datetime.now().strftime("pcr_%Y%m%d_%H%M%S")
    run_dir = result_root / run_name
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}. Use --overwrite or a new --run-name.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_validate(bundle: dict, run_dir: Path) -> dict:
    dataset = load_dataset(bundle["dataset"])
    validation = validate_dataset(dataset)
    write_json(run_dir / "validation.json", validation)
    return validation


def run_pairwise(bundle: dict, run_dir: Path) -> dict:
    dataset = load_dataset(bundle["dataset"])
    if not dataset.steps:
        raise ValueError("Dataset has no walk-forward steps.")
    step = dataset.steps[0]
    algorithm_config = bundle["algorithm_config"]
    register = get_algorithm(str(algorithm_config["name"]))
    result = register(step.source, dataset.initial_world, algorithm_config)
    source_raw = read_point_cloud(step.source)
    target_raw = read_point_cloud(dataset.initial_world)
    step_dir = run_dir / "steps" / step.name
    save_pair_outputs(step_dir, result, source_raw, target_raw)
    summary = {
        "mode": "pairwise",
        "step": step.name,
        "source": str(step.source),
        "target": str(dataset.initial_world),
        "status": result.status,
        "metrics": result.metrics,
    }
    write_json(run_dir / "summary.json", summary)
    write_summary_md(run_dir / "summary.md", [summary])
    return summary


def run_walk_forward(bundle: dict, run_dir: Path) -> dict:
    dataset = load_dataset(bundle["dataset"])
    algorithm_config = bundle["algorithm_config"]
    fusion_config = algorithm_config.get("fusion", {})
    register = get_algorithm(str(algorithm_config["name"]))
    current_world = read_point_cloud(dataset.initial_world)
    current_world_path = dataset.initial_world
    step_summaries = []

    for index, step in enumerate(dataset.steps, 1):
        step_dir = run_dir / "steps" / f"{index:02d}_{step.name}"
        result = register(step.source, current_world_path, algorithm_config)
        source_raw = read_point_cloud(step.source)
        target_raw = read_point_cloud(current_world_path)
        save_pair_outputs(step_dir, result, source_raw, target_raw)

        fused = fuse_world(
            target_raw,
            source_raw,
            result.matrix,
            fusion_voxel_size=float(fusion_config.get("fusion_voxel_size", 0.06)),
            sor_enabled=bool(fusion_config.get("fusion_sor_enabled", False)),
            sor_nb_neighbors=int(fusion_config.get("fusion_sor_nb_neighbors", 20)),
            sor_std_ratio=float(fusion_config.get("fusion_sor_std_ratio", 2.0)),
        )
        fused_path = step_dir / "fused_world.ply"
        write_point_cloud(fused_path, fused)

        current_world = fused
        current_world_path = fused_path
        step_summary = {
            "mode": "walk_forward",
            "step_index": index,
            "step": step.name,
            "source": str(step.source),
            "target": str(result.target_path),
            "output_world": str(fused_path),
            "status": result.status,
            "metrics": result.metrics,
            "fused_points": len(fused.points),
        }
        step_summaries.append(step_summary)

    final_path = run_dir / "final_world.ply"
    write_point_cloud(final_path, current_world)
    summary = {
        "mode": "walk_forward",
        "dataset": dataset.name,
        "algorithm": algorithm_config["name"],
        "step_count": len(step_summaries),
        "final_world": str(final_path),
        "steps": step_summaries,
    }
    write_json(run_dir / "summary.json", summary)
    write_summary_md(run_dir / "summary.md", step_summaries)
    return summary


def write_summary_md(path: Path, steps: list[dict]) -> None:
    lines = [
        "| step | status | fitness | trimmed | scale_min | scale_max | fused_points |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in steps:
        metrics = item.get("metrics", {})
        lines.append(
            "| {step} | {status} | {fitness:.6f} | {trimmed} | {scale_min:.3f} | {scale_max:.3f} | {points} |".format(
                step=item.get("step", ""),
                status=item.get("status", ""),
                fitness=float(metrics.get("eval_fitness") or 0.0),
                trimmed="None"
                if metrics.get("eval_trimmed_mean_nn_dist") is None
                else f"{float(metrics['eval_trimmed_mean_nn_dist']):.6f}",
                scale_min=float(metrics.get("scale_min") or 0.0),
                scale_max=float(metrics.get("scale_max") or 0.0),
                points=item.get("fused_points", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCR validation, pairwise registration, or walk-forward fusion.")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--mode", choices=["validate", "pairwise", "walk_forward"], default="walk_forward")
    parser.add_argument("--result-root", type=Path, default=Path("result"))
    parser.add_argument("--run-name")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_experiment(args.experiment)
    run_dir = make_run_dir(args.result_root, args.run_name, args.overwrite)
    write_yaml(run_dir / "run_config.yaml", bundle)
    if args.mode == "validate":
        summary = run_validate(bundle, run_dir)
    elif args.mode == "pairwise":
        summary = run_pairwise(bundle, run_dir)
    else:
        summary = run_walk_forward(bundle, run_dir)
    print(f"Run directory: {run_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()

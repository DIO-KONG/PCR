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
from utils.fusion import fuse_world_with_stats
from utils.io import save_pair_outputs, write_json, write_point_cloud, write_yaml
from utils.pointcloud import read_point_cloud


DEFAULT_EXPERIMENT = Path("testbench/configs/experiments/basic_walk_forward.yaml")


def load_experiment(path: Path) -> dict:
    """读取实验配置，并把 dataset 与 algorithm 配置展开为一个 bundle。

    这样 testbench 的运行逻辑只需要处理一个完整配置对象，不必在每个模式里重复读取 YAML。
    `algorithm_overrides` 用于局部覆盖算法参数，适合快速做参数实验。
    """

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
    """创建本次实验的输出目录。

    默认使用时间戳作为 run name；如果用户显式指定 `--run-name`，除非加 `--overwrite`，
    否则不会覆盖已有目录，避免不小心冲掉之前的可视化结果。
    """

    if run_name is None:
        run_name = datetime.now().strftime("pcr_%Y%m%d_%H%M%S")
    run_dir = result_root / run_name
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}. Use --overwrite or a new --run-name.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_validate(bundle: dict, run_dir: Path) -> dict:
    """只验证数据集，不执行配准。

    这个模式用于快速确认配置中的 PLY 都存在、可读、点数非零。
    """

    dataset = load_dataset(bundle["dataset"])
    validation = validate_dataset(dataset)
    write_json(run_dir / "validation.json", validation)
    return validation


def run_pairwise(bundle: dict, run_dir: Path) -> dict:
    """执行数据集第一步 pairwise 配准。

    这是最小烟测：只看第一个 incremental 是否能配准到 baseline/world。
    """

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
    """执行完整 walk-forward 配准和融合。

    每一步都把当前 source 配准到当前累计 world，然后按照配置中的融合策略生成下一步 world。
    当前支持普通追加融合和“重影检测 / 冲突过滤”融合；后者会额外保存 accepted/conflict/duplicate
    调试点云，便于人工观察哪些点被加入，哪些点被当作重影丢弃。
    """

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

        fusion_result = fuse_world_with_stats(
            target_raw,
            source_raw,
            result.matrix,
            fusion_voxel_size=float(fusion_config.get("fusion_voxel_size", 0.06)),
            method=str(fusion_config.get("fusion_method", "simple")),
            duplicate_distance=float(fusion_config.get("duplicate_distance", 0.08)),
            conflict_distance=float(fusion_config.get("conflict_distance", 0.20)),
            sor_enabled=bool(fusion_config.get("fusion_sor_enabled", False)),
            sor_nb_neighbors=int(fusion_config.get("fusion_sor_nb_neighbors", 20)),
            sor_std_ratio=float(fusion_config.get("fusion_sor_std_ratio", 2.0)),
        )
        fused = fusion_result.cloud
        fused_path = step_dir / "fused_world.ply"
        write_point_cloud(fused_path, fused)

        # 调试点云默认只在冲突过滤实验中保存。PLY 已被 .gitignore 忽略，
        # 因此这些文件可以放心用于可视化排查，不会污染版本库。
        if bool(fusion_config.get("output_debug_clouds", False)):
            debug_dir = step_dir / "fusion_debug"
            for name, cloud in fusion_result.debug_clouds.items():
                if len(cloud.points):
                    write_point_cloud(debug_dir / f"{name}.ply", cloud)
        write_json(step_dir / "fusion_stats.json", fusion_result.stats)

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
            "fusion_stats": fusion_result.stats,
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


def run_fusion_sweep(bundle: dict, run_dir: Path) -> dict:
    """针对多组融合阈值重复运行 walk-forward。

    设计目的：配准算法保持一致，只改变融合阶段的 duplicate/conflict 阈值，
    最后把每组 `final_world.ply` 都保留下来，方便用 Open3D/CloudCompare 做视觉择优。
    """

    experiment = bundle["experiment"]
    sweeps = experiment.get("fusion_sweeps", [])
    if not sweeps:
        raise ValueError("fusion_sweep mode requires `fusion_sweeps` in the experiment config.")

    variant_summaries = []
    for index, sweep in enumerate(sweeps, 1):
        name = str(sweep.get("name", f"variant_{index:02d}"))
        variant_dir = run_dir / f"{index:02d}_{name}"

        # 每个 sweep 只覆盖算法配置中的 fusion 字段，其余配准参数保持一致。
        variant_bundle = dict(bundle)
        variant_bundle["algorithm_config"] = deep_merge(
            bundle["algorithm_config"],
            {"fusion": {key: value for key, value in sweep.items() if key != "name"}},
        )
        write_yaml(variant_dir / "run_config.yaml", variant_bundle)
        summary = run_walk_forward(variant_bundle, variant_dir)
        summary["sweep_name"] = name
        summary["sweep_config"] = sweep
        variant_summaries.append(summary)

    sweep_summary = {
        "mode": "fusion_sweep",
        "variant_count": len(variant_summaries),
        "variants": [
            {
                "sweep_name": item["sweep_name"],
                "final_world": item["final_world"],
                "step_count": item["step_count"],
            }
            for item in variant_summaries
        ],
    }
    write_json(run_dir / "summary.json", sweep_summary)
    write_sweep_summary_md(run_dir / "summary.md", variant_summaries)
    return sweep_summary


def write_summary_md(path: Path, steps: list[dict]) -> None:
    """写出单次 walk-forward 的 Markdown 摘要表。

    摘要表只放最关键指标，完整指标仍然保存在 `summary.json` 和每步 `metrics.json`。
    """

    lines = [
        "| step | status | fitness | trimmed | scale_min | scale_max | accepted | duplicate | conflict | fused_points |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in steps:
        metrics = item.get("metrics", {})
        fusion_stats = item.get("fusion_stats", {})
        lines.append(
            "| {step} | {status} | {fitness:.6f} | {trimmed} | {scale_min:.3f} | {scale_max:.3f} | {accepted} | {duplicate} | {conflict} | {points} |".format(
                step=item.get("step", ""),
                status=item.get("status", ""),
                fitness=float(metrics.get("eval_fitness") or 0.0),
                trimmed="None"
                if metrics.get("eval_trimmed_mean_nn_dist") is None
                else f"{float(metrics['eval_trimmed_mean_nn_dist']):.6f}",
                scale_min=float(metrics.get("scale_min") or 0.0),
                scale_max=float(metrics.get("scale_max") or 0.0),
                accepted=fusion_stats.get("accepted_new_points", ""),
                duplicate=fusion_stats.get("duplicate_points", ""),
                conflict=fusion_stats.get("conflict_points", ""),
                points=item.get("fused_points", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sweep_summary_md(path: Path, variants: list[dict]) -> None:
    """写出多组融合参数 sweep 的总览表。"""

    lines = [
        "| variant | final_world | final_points | statuses |",
        "|---|---|---:|---|",
    ]
    for item in variants:
        final_step = item["steps"][-1] if item.get("steps") else {}
        statuses = ",".join(step.get("status", "") for step in item.get("steps", []))
        lines.append(
            "| {name} | {final_world} | {points} | {statuses} |".format(
                name=item["sweep_name"],
                final_world=item["final_world"],
                points=final_step.get("fused_points", ""),
                statuses=statuses,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCR validation, pairwise registration, or walk-forward fusion.")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--mode",
        choices=["validate", "pairwise", "walk_forward", "fusion_sweep"],
        default="walk_forward",
    )
    parser.add_argument("--result-root", type=Path, default=Path("result"))
    parser.add_argument("--run-name")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = parse_args()
    bundle = load_experiment(args.experiment)
    run_dir = make_run_dir(args.result_root, args.run_name, args.overwrite)
    write_yaml(run_dir / "run_config.yaml", bundle)
    if args.mode == "validate":
        summary = run_validate(bundle, run_dir)
    elif args.mode == "pairwise":
        summary = run_pairwise(bundle, run_dir)
    elif args.mode == "fusion_sweep":
        summary = run_fusion_sweep(bundle, run_dir)
    else:
        summary = run_walk_forward(bundle, run_dir)
    print(f"Run directory: {run_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()

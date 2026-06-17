#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.base import RegistrationCandidate
from algorithm.shared_frame_alignment import (
    collect_correspondences,
    evaluate_shared_correspondences,
    image_name_list,
    load_npz,
    ransac_rigid_candidates,
    rotation_angle_deg,
)
from utils.pointcloud import load_cloud, make_overlay, save_cloud, transform_cloud
from utils.preprocess import (
    describe_point_cloud,
    json_safe as preprocess_json_safe,
    load_point_cloud,
    preprocess_for_registration,
    save_point_cloud,
    write_debug_outputs,
)


DEFAULT_CONFIG = Path("testbench/configs/experiments/shared_frame_walkforward.yaml")


def json_safe(value: Any) -> Any:
    """把 numpy、Path、dataclass 转成 JSON 可写类型。"""

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


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    """写 JSON 文件，自动创建父目录。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_yaml(path: str | Path, payload: Any) -> None:
    """写 YAML 配置快照。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(json_safe(payload), allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_matrix(path: str | Path, matrix: np.ndarray) -> None:
    """写 4x4 齐次矩阵。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(matrix, dtype=float), fmt="%.10f")


def frame_sort_key(name: str) -> tuple[int, str]:
    """按图像文件名前缀数字排序，例如 `10-rgb.png` 应排在 `9-rgb.png` 后。"""

    match = re.match(r"^(\d+)", str(name))
    if match:
        return int(match.group(1)), str(name)
    return 10**9, str(name)


def discover_shared_frames(source_batch: dict[str, np.ndarray], target_batch: dict[str, np.ndarray]) -> list[str]:
    """自动发现两个 DA3 batch 的共享 RGB 帧。"""

    source_names = set(image_name_list(source_batch))
    target_names = set(image_name_list(target_batch))
    return sorted(source_names & target_names, key=frame_sort_key)


def ensure_floor_removed(raw_path: str | Path, output_path: str | Path) -> Path:
    """确保某个 raw PLY 已生成地板对齐后的去地板点云。

    本函数只在输出缺失时运行预处理，避免重复覆盖已经人工检查过的结果。
    """

    output_path = Path(output_path)
    if output_path.exists():
        cloud = load_cloud(output_path)
        if not cloud.is_empty():
            return output_path

    raw_path = Path(raw_path)
    print(f"[preprocess] {raw_path} -> {output_path}", flush=True)
    cloud = load_point_cloud(raw_path)
    result = preprocess_for_registration(cloud, align_floor=True, remove_floor=True)
    save_point_cloud(output_path, result.cloud)
    write_debug_outputs(output_path.parent / "debug", result)
    return output_path


def make_algorithm_config(base: dict[str, Any], source_npz: str, target_npz: str, shared_frames: list[str] | None = None) -> dict[str, Any]:
    """构造 shared-frame 算法所需配置。

    这里复用 `algorithm.shared_frame_alignment` 的底层函数，只把 walk-forward
    配置中的参数映射成它们需要的键。
    """

    algorithm = base["algorithm"]
    config = {
        "source_npz": source_npz,
        "target_npz": target_npz,
        "sampling": dict(algorithm["sampling"]),
        "ransac": dict(algorithm["ransac"]),
        "refinement": dict(algorithm["refinement"]),
        "icp_refinement": {"enabled": False},
    }
    if shared_frames is not None:
        config["shared_frames"] = list(shared_frames)
    return config


def rank_frame_combinations(
    source_npz: str | Path,
    target_npz: str | Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """枚举共享帧 4选3，返回排序结果和最佳粗配准候选。

    如果未来某步共享帧多于 4 帧，仍按配置的 combo_size 组合选择；如果刚好
    只有 3 帧，则只有一个组合。
    """

    source_batch = load_npz(source_npz)
    target_batch = load_npz(target_npz)
    shared_frames = discover_shared_frames(source_batch, target_batch)
    combo_size = int(config["algorithm"].get("frame_combo_size", 3))
    if len(shared_frames) < combo_size:
        raise RuntimeError(f"Not enough shared frames: shared={shared_frames}, combo_size={combo_size}")

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for frame_combo in itertools.combinations(shared_frames, combo_size):
        algo_config = make_algorithm_config(config, str(source_npz), str(target_npz), list(frame_combo))
        source_points, target_points, frame_names, frame_stats = collect_correspondences(source_batch, target_batch, algo_config)
        candidate = ransac_rigid_candidates(source_points, target_points, frame_names, algo_config)[0]
        row = {
            "frames": list(frame_combo),
            "score": float(candidate.score),
            "correspondence_count": int(len(source_points)),
            "coarse_threshold": float(candidate.metrics.get("coarse_threshold", 0.0)),
            "inlier_ratio": float(candidate.metrics.get("inlier_ratio", 0.0)),
            "inlier_count": int(candidate.metrics.get("inlier_count", 0)),
            "median_error": float(candidate.metrics.get("median_error", float("inf"))),
            "p90_error": float(candidate.metrics.get("p90_error", float("inf"))),
            "rmse": float(candidate.metrics.get("rmse", float("inf"))),
            "per_frame_metrics": candidate.metrics.get("per_frame_metrics", []),
        }
        rows.append(row)
        payload = {
            "row": row,
            "candidate": candidate,
            "source_points": source_points,
            "target_points": target_points,
            "frame_names": frame_names,
            "frame_stats": frame_stats,
            "shared_frames": shared_frames,
        }
        if best is None or row["score"] > best["row"]["score"]:
            best = payload

    if best is None:
        raise RuntimeError("No valid frame-combination candidate was produced.")
    return sorted(rows, key=lambda item: item["score"], reverse=True), best


def run_bounded_icp(
    source_cloud: o3d.geometry.PointCloud,
    target_world: o3d.geometry.PointCloud,
    initial_matrix: np.ndarray,
    icp_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """以 coarse global 矩阵为初值，对累计 world 做有边界 point-to-plane ICP。"""

    voxel_size = float(icp_config["voxel_size"])
    source_down = source_cloud.voxel_down_sample(voxel_size)
    target_down = target_world.voxel_down_sample(voxel_size)
    if source_down.is_empty() or target_down.is_empty():
        raise RuntimeError("ICP downsampled source or target world is empty.")

    search = o3d.geometry.KDTreeSearchParamHybrid(
        radius=float(icp_config["normal_radius"]),
        max_nn=int(icp_config["normal_max_nn"]),
    )
    source_down.estimate_normals(search)
    target_down.estimate_normals(search)

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        float(icp_config["max_correspondence_distance"]),
        np.asarray(initial_matrix, dtype=float),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(icp_config["max_iteration"])),
    )
    icp_matrix = np.asarray(result.transformation, dtype=float)
    delta = icp_matrix @ np.linalg.inv(initial_matrix)
    translation_delta = float(np.linalg.norm(delta[:3, 3]))
    rotation_delta = rotation_angle_deg(delta[:3, :3])
    accepted = (
        translation_delta <= float(icp_config["max_translation_delta"])
        and rotation_delta <= float(icp_config["max_rotation_delta_deg"])
    )
    metrics = {
        "status": "accepted" if accepted else "rejected_by_boundary",
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "translation_delta": translation_delta,
        "rotation_delta_deg": rotation_delta,
        "max_translation_delta": float(icp_config["max_translation_delta"]),
        "max_rotation_delta_deg": float(icp_config["max_rotation_delta_deg"]),
        "max_correspondence_distance": float(icp_config["max_correspondence_distance"]),
        "voxel_size": voxel_size,
    }
    return icp_matrix, delta, metrics


def save_frame_combo_ranking(path: Path, rows: list[dict[str, Any]]) -> None:
    """保存每步动态选帧候选表。"""

    lines = [
        "| rank | frames | score | corr | inlier_ratio | median_m | p90_m | rmse_m |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, 1):
        lines.append(
            "| {rank} | {frames} | {score:.6f} | {corr} | {ratio:.6f} | {median:.6f} | {p90:.6f} | {rmse:.6f} |".format(
                rank=rank,
                frames=", ".join(row["frames"]),
                score=row["score"],
                corr=row["correspondence_count"],
                ratio=row["inlier_ratio"],
                median=row["median_error"],
                p90=row["p90_error"],
                rmse=row["rmse"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_summary_row(lines: list[str], row: dict[str, Any]) -> None:
    """向 summary Markdown 添加一步结果。"""

    lines.append(
        "| {step} | {source} -> {target} | {frames} | {coarse_median:.4f} | {icp_status} | {icp_fit:.4f} | {icp_rmse:.4f} | {dt:.4f} | {dr:.3f} | {used} | {points} |".format(
            step=row["step_index"],
            source=row["source_id"],
            target=row["target_id"],
            frames=", ".join(row["selected_frames"]),
            coarse_median=row["coarse_metrics"]["median_error"],
            icp_status=row["icp_metrics"]["status"],
            icp_fit=row["icp_metrics"]["fitness"],
            icp_rmse=row["icp_metrics"]["inlier_rmse"],
            dt=row["icp_metrics"]["translation_delta"],
            dr=row["icp_metrics"]["rotation_delta_deg"],
            used=row["final_source"],
            points=row["fused_world_points"],
        )
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """执行完整 dynamic-top3 shared-frame walk-forward 建图实验。"""

    config = load_yaml(args.config)
    run_name = args.run_name or datetime.now().strftime("walkforward_%Y%m%d_%H%M%S")
    result_root = Path(config.get("result_root", "result/walkforward_shared_frame"))
    run_dir = result_root / run_name
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Run directory exists: {run_dir}. Use --overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "transforms").mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "run_config.yaml", config)

    baseline = config["baseline"]
    baseline_cloud_path = ensure_floor_removed(baseline["raw_cloud"], baseline["cloud"])
    world = load_cloud(baseline_cloud_path)
    transforms: dict[str, np.ndarray] = {baseline["id"]: np.eye(4)}
    write_matrix(run_dir / "transforms" / f"{baseline['id']}_to_global.txt", np.eye(4))

    algorithm_record = (
        "# dynamic_top3_shared_frame_bounded_icp\n\n"
        "- 自动发现相邻 DA3 batch 共享帧，并枚举 4选3 粗配准。\n"
        "- 粗配准使用共享像素 3D 对应点、RANSAC、Kabsch/SVD，只估计刚体。\n"
        "- source 投到 global 后，以累计 world 为 target 做 bounded point-to-plane ICP。\n"
        "- ICP 未超过平移/旋转边界则采用；共享帧一致性只作为诊断。\n"
    )
    (run_dir / "algorithm_record.md").write_text(algorithm_record, encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    summary_lines = [
        "# Shared-Frame Dynamic Top3 Walk-Forward Summary",
        "",
        "| step | pair | selected frames | coarse median m | icp status | icp fitness | icp rmse m | icp dt m | icp drot deg | final | fused points |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---|---:|",
    ]

    for step_index, step in enumerate(config["steps"], 1):
        print(f"[step {step_index}] {step['source_id']} -> {step['target_id']}", flush=True)
        step_dir = run_dir / f"step_{step_index:02d}_{step['source_id']}_to_{step['target_id']}"
        step_dir.mkdir(parents=True, exist_ok=True)

        source_cloud_path = ensure_floor_removed(step["source_raw_cloud"], step["source_cloud"])
        ensure_floor_removed(step["target_raw_cloud"], step["target_cloud"])
        source_cloud = load_cloud(source_cloud_path)

        combo_rows, best = rank_frame_combinations(step["source_npz"], step["target_npz"], config)
        pairwise_matrix = np.asarray(best["candidate"].matrix, dtype=float)
        target_to_global = transforms[step["target_id"]]
        coarse_global = target_to_global @ pairwise_matrix
        coarse_registered = transform_cloud(source_cloud, coarse_global)
        coarse_overlay = make_overlay(world, coarse_registered)

        icp_matrix, icp_delta, icp_metrics = run_bounded_icp(source_cloud, world, coarse_global, config["algorithm"]["icp"])
        final_matrix = icp_matrix if icp_metrics["status"] == "accepted" else coarse_global
        final_source = "bounded_icp" if icp_metrics["status"] == "accepted" else "coarse"
        bounded_registered = transform_cloud(source_cloud, icp_matrix)
        bounded_overlay = make_overlay(world, bounded_registered)
        final_registered = transform_cloud(source_cloud, final_matrix)

        fused_world = world + final_registered
        fused_world = fused_world.voxel_down_sample(float(config["algorithm"]["fusion"]["voxel_size"]))
        world = fused_world
        transforms[step["source_id"]] = final_matrix

        coarse_shared_metrics = evaluate_shared_correspondences(
            best["source_points"],
            best["target_points"],
            best["frame_names"],
            pairwise_matrix,
            float(config["algorithm"]["ransac"]["evaluation_threshold"]),
        )

        save_cloud(step_dir / "coarse_overlay.ply", coarse_overlay)
        save_cloud(step_dir / "bounded_icp_overlay.ply", bounded_overlay)
        save_cloud(step_dir / "final_registered_source.ply", final_registered)
        save_cloud(step_dir / "fused_world.ply", fused_world)
        write_matrix(step_dir / "coarse_matrix.txt", coarse_global)
        write_matrix(step_dir / "bounded_icp_matrix.txt", icp_matrix)
        write_matrix(step_dir / "bounded_icp_delta_from_coarse.txt", icp_delta)
        write_matrix(step_dir / "final_matrix.txt", final_matrix)
        write_matrix(run_dir / "transforms" / f"{step['source_id']}_to_global.txt", final_matrix)
        write_json(
            step_dir / "selected_frames.json",
            {
                "shared_frames_discovered": best["shared_frames"],
                "selected_frames": best["row"]["frames"],
                "selection": best["row"],
            },
        )
        save_frame_combo_ranking(step_dir / "frame_combo_ranking.md", combo_rows)

        step_metrics = {
            "step_index": step_index,
            "step": step,
            "selected_frames": best["row"]["frames"],
            "shared_frames_discovered": best["shared_frames"],
            "frame_combo_ranking": combo_rows,
            "pairwise_matrix_source_to_target": pairwise_matrix,
            "target_to_global": target_to_global,
            "coarse_matrix_source_to_global": coarse_global,
            "bounded_icp_matrix_source_to_global": icp_matrix,
            "bounded_icp_delta_from_coarse": icp_delta,
            "final_matrix_source_to_global": final_matrix,
            "final_source": final_source,
            "coarse_metrics": coarse_shared_metrics,
            "icp_metrics": icp_metrics,
            "source_cloud_points": int(len(source_cloud.points)),
            "fused_world_points": int(len(fused_world.points)),
            "world_stats": describe_point_cloud(fused_world),
        }
        write_json(step_dir / "metrics.json", step_metrics)
        summary_row = {
            "step_index": step_index,
            "source_id": step["source_id"],
            "target_id": step["target_id"],
            "selected_frames": best["row"]["frames"],
            "coarse_metrics": coarse_shared_metrics,
            "icp_metrics": icp_metrics,
            "final_source": final_source,
            "fused_world_points": int(len(fused_world.points)),
        }
        summary_rows.append(summary_row)
        append_summary_row(summary_lines, summary_row)
        print(
            "[step {idx}] frames={frames} coarse_median={median:.4f} icp={status} dt={dt:.4f} drot={dr:.3f} world_points={pts}".format(
                idx=step_index,
                frames=",".join(best["row"]["frames"]),
                median=coarse_shared_metrics["median_error"],
                status=icp_metrics["status"],
                dt=icp_metrics["translation_delta"],
                dr=icp_metrics["rotation_delta_deg"],
                pts=len(fused_world.points),
            ),
            flush=True,
        )

    save_cloud(run_dir / "final_world.ply", world)
    write_json(
        run_dir / "summary.json",
        {
            "run_name": run_name,
            "run_dir": run_dir,
            "summary_rows": summary_rows,
            "final_world_points": int(len(world.points)),
            "transforms": transforms,
        },
    )
    (run_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {"run_dir": str(run_dir), "final_world_points": int(len(world.points)), "steps": summary_rows}


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run shared-frame dynamic-top3 walk-forward registration.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(json_safe(run(parse_args())), indent=2, ensure_ascii=False))

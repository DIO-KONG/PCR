from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def batch_stem(index: int, label: str, first: int, last: int) -> str:
    """生成和 DA3 runner 一致的 batch stem。"""

    return f"batch_{index:03d}_{label}_{first:03d}-{last:03d}"


def batch_ref(stem: str, *, preprocess_mode: str) -> dict[str, str]:
    """生成配置中的 batch 文件路径。"""

    if preprocess_mode == "floor_removed":
        cloud_path = f"result/preprocess/{stem}/floor_removed.ply"
    elif preprocess_mode == "align_only":
        cloud_path = f"result/preprocess_align_only/{stem}/aligned.ply"
    else:
        raise ValueError(f"Unsupported preprocess mode: {preprocess_mode}")

    return {
        "npz": f"da3/data/raw/pointcloud/{stem}.npz",
        "raw_cloud": f"da3/data/raw/pointcloud/{stem}.ply",
        "cloud": cloud_path,
    }


def build_config(
    *,
    frame_start: int,
    frame_end: int,
    baseline_size: int,
    window_size: int,
    preprocess_mode: str,
    icp_method: str,
    max_scale_delta: float,
) -> dict[str, Any]:
    """生成 online submap walk-forward YAML 内容。"""

    baseline_last = frame_start + baseline_size - 1
    baseline_stem = batch_stem(1, "baseline", frame_start, baseline_last)
    baseline = {"id": baseline_stem, **batch_ref(baseline_stem, preprocess_mode=preprocess_mode)}

    steps: list[dict[str, Any]] = []
    previous_stem = baseline_stem
    batch_index = 2
    step_index = 1
    for current_frame in range(baseline_last + 1, frame_end + 1):
        first = current_frame - window_size + 1
        source_stem = batch_stem(batch_index, "window", first, current_frame)
        source = batch_ref(source_stem, preprocess_mode=preprocess_mode)
        target = batch_ref(previous_stem, preprocess_mode=preprocess_mode)
        steps.append(
            {
                "id": f"step_{step_index:03d}_{source_stem}_to_{previous_stem}",
                "source_id": source_stem,
                "target_id": previous_stem,
                "source_npz": source["npz"],
                "target_npz": target["npz"],
                "source_raw_cloud": source["raw_cloud"],
                "target_raw_cloud": target["raw_cloud"],
                "source_cloud": source["cloud"],
                "target_cloud": target["cloud"],
            }
        )
        previous_stem = source_stem
        batch_index += 1
        step_index += 1

    remove_floor = preprocess_mode == "floor_removed"
    icp_config: dict[str, Any] = {
        "voxel_size": 0.06,
        "max_correspondence_distance": 0.08,
        "normal_radius": 0.18,
        "normal_max_nn": 30,
        "max_iteration": 40,
        "max_translation_delta": 0.15,
        "max_rotation_delta_deg": 5.0,
        "method": icp_method,
    }
    if icp_method == "sim3_point_to_point":
        icp_config.update(
            {
                "max_scale_delta": float(max_scale_delta),
                "min_correspondences": 80,
                "sim3_convergence_rmse_delta": 1.0e-5,
            }
        )

    return {
        "name": "online_submap_walkforward_327",
        "result_root": "result/submap_walkforward",
        "baseline": baseline,
        "steps": steps,
        "preprocess": {
            "align_floor": True,
            "remove_floor": remove_floor,
        },
        "submap": {
            "submap_size": 10,
            "submap_overlap": 3,
            "fusion_mode": "new_frame_only",
        },
        "quality_gate": {
            "max_shared_median_error": 0.06,
            "max_shared_p90_error": 0.12,
            "max_icp_rmse": 0.08,
            "min_icp_fitness": 0.35,
            "max_icp_translation_delta": 0.15,
            "max_icp_rotation_delta_deg": 5.0,
            "max_conflict_ratio": 0.30,
        },
        "fusion": {
            "voxel_size": 0.06,
            "duplicate_distance": 0.04,
            "conflict_distance": 0.15,
            "normal_angle_deg": 25.0,
            "normal_radius": 0.18,
            "normal_max_nn": 30,
            "conf_percentile": 20.0,
            "frame_point_stride": 1,
            "max_points_per_frame": 120000,
            "random_seed": 7,
            "floor_distance_threshold": 0.03,
            "global_preview_voxel_size": 0.08,
        },
        "algorithm": {
            "frame_combo_size": 3,
            "sampling": {
                "conf_percentile": 20.0,
                "stride": 6,
                "max_points_per_frame": 8000,
                "random_seed": 7,
            },
            "ransac": {
                "iterations": 3000,
                "sample_size": 6,
                "thresholds": [0.18, 0.14, 0.10, 0.06],
                "evaluation_threshold": 0.06,
                "min_inliers": 50,
                "seed": 7,
            },
            "refinement": {
                "enabled": True,
                "thresholds": [0.18, 0.14, 0.10, 0.06],
                "iterations_per_threshold": 2,
                "min_points": 200,
            },
            "icp": {
                **icp_config,
            },
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explicit online submap walk-forward YAML.")
    parser.add_argument("--output", type=Path, default=Path("testbench/configs/experiments/submap_walkforward_327.yaml"))
    parser.add_argument("--frame-start", type=int, default=1)
    parser.add_argument("--frame-end", type=int, default=327)
    parser.add_argument("--baseline-size", type=int, default=12)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument(
        "--preprocess-mode",
        choices=("floor_removed", "align_only"),
        default="floor_removed",
        help="floor_removed removes the detected floor; align_only keeps floor points after floor alignment.",
    )
    parser.add_argument(
        "--icp-method",
        choices=("point_to_plane_icp", "sim3_point_to_point"),
        default="point_to_plane_icp",
    )
    parser.add_argument("--max-scale-delta", type=float, default=0.08)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        baseline_size=args.baseline_size,
        window_size=args.window_size,
        preprocess_mode=args.preprocess_mode,
        icp_method=args.icp_method,
        max_scale_delta=args.max_scale_delta,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.output} with {len(config['steps'])} steps")


if __name__ == "__main__":
    main()

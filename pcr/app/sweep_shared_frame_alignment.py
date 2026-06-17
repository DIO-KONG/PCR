from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pcr.algorithms.shared_frame.coarse import ransac_rigid_candidates
from pcr.algorithms.shared_frame.correspondences import collect_correspondences
from pcr.algorithms.shared_frame.frame_selection import discover_shared_frames
from pcr.artifacts.store import json_safe
from pcr.config.loader import load_yaml
from pcr.io.da3_npz import load_da3_batch


DEFAULT_ALGORITHM_CONFIG = Path("testbench/configs/algorithms/shared_frame_alignment.yaml")
DEFAULT_RESULT_ROOT = Path("result/shared_frame_sweep")


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def make_frame_sets(shared_frames: list[str], combo_size: int, include_all: bool) -> list[tuple[str, ...]]:
    """生成共享帧组合，复用正式算法的 shared-frame discovery。"""

    if len(shared_frames) < combo_size:
        raise ValueError(f"Not enough shared frames for {combo_size}-choose-k sweep: {shared_frames}")
    frame_sets = list(itertools.combinations(shared_frames, combo_size))
    if include_all:
        frame_sets.append(tuple(shared_frames))
    return frame_sets


def candidate_row(run_id: int, config: dict[str, Any], frame_set: tuple[str, ...], candidate, correspondence_count: int) -> dict[str, Any]:
    """把一个候选压成 CSV/Markdown 友好的扁平行。"""

    metrics = candidate.metrics.to_dict()
    return {
        "run_id": run_id,
        "score": float(candidate.score),
        "candidate_id": int(candidate.candidate_id),
        "conf_percentile": float(config["sampling"]["conf_percentile"]),
        "stride": int(config["sampling"]["stride"]),
        "frames": ",".join(frame_set),
        "frame_count": len(frame_set),
        "correspondence_count": int(correspondence_count),
        "coarse_threshold": float(metrics.get("coarse_threshold", 0.0)),
        "inlier_ratio": float(metrics.get("inlier_ratio", 0.0)),
        "inlier_count": int(metrics.get("inlier_count", 0)),
        "median_error": float(metrics.get("median_error", float("inf"))),
        "p90_error": float(metrics.get("p90_error", float("inf"))),
        "rmse": float(metrics.get("rmse", float("inf"))),
        "mean_error": float(metrics.get("mean_error", float("inf"))),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], top_n: int) -> None:
    lines = [
        "# Shared Frame Sweep Summary",
        "",
        f"Top {min(top_n, len(rows))} candidates sorted by score.",
        "",
        "| rank | run_id | score | conf_drop | stride | frames | corr | coarse_th | inlier_ratio | median_m | p90_m | rmse_m |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows[:top_n], 1):
        lines.append(
            "| {rank} | {run_id} | {score:.6f} | {conf:.1f} | {stride} | {frames} | {corr} | {coarse:.3f} | {ratio:.6f} | {median:.6f} | {p90:.6f} | {rmse:.6f} |".format(
                rank=rank,
                run_id=row["run_id"],
                score=row["score"],
                conf=row["conf_percentile"],
                stride=row["stride"],
                frames=row["frames"],
                corr=row["correspondence_count"],
                coarse=row["coarse_threshold"],
                ratio=row["inlier_ratio"],
                median=row["median_error"],
                p90=row["p90_error"],
                rmse=row["rmse"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """执行共享帧粗配准参数 sweep。"""

    base_config = load_yaml(args.config)
    source_batch = load_da3_batch(base_config["source_npz"])
    target_batch = load_da3_batch(base_config["target_npz"])
    shared_frames = discover_shared_frames(source_batch, target_batch)
    frame_sets = make_frame_sets(shared_frames, args.frame_combo_size, args.include_all_frames)

    run_name = args.run_name or datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
    run_dir = args.result_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    run_id = 0

    for conf_percentile, stride, frame_set in itertools.product(args.conf_percentiles, args.strides, frame_sets):
        run_id += 1
        config = copy.deepcopy(base_config)
        config["shared_frames"] = list(frame_set)
        config.setdefault("sampling", {})
        config["sampling"]["conf_percentile"] = float(conf_percentile)
        config["sampling"]["stride"] = int(stride)
        config.setdefault("ransac", {})
        config["ransac"]["thresholds"] = [float(item) for item in args.ransac_thresholds]
        config["ransac"]["iterations"] = int(args.ransac_iterations)
        config.setdefault("refinement", {})
        config["refinement"]["thresholds"] = [float(item) for item in args.ransac_thresholds]
        config.setdefault("icp_refinement", {})
        config["icp_refinement"]["enabled"] = False

        try:
            source_points, target_points, frame_names, frame_stats = collect_correspondences(source_batch, target_batch, config)
            candidates = ransac_rigid_candidates(
                source_points,
                target_points,
                frame_names,
                config,
                source_frame="source",
                target_frame="target",
            )
            best = candidates[0]
            row = candidate_row(run_id, config, frame_set, best, len(source_points))
            row["status"] = "ok"
            all_rows.append(row)
            detailed.append(
                {
                    "run_id": run_id,
                    "status": "ok",
                    "config": config,
                    "frame_stats": frame_stats,
                    "best_candidate": best,
                }
            )
        except Exception as exc:  # noqa: BLE001 - 参数 sweep 需要保留失败组合继续跑。
            row = {
                "run_id": run_id,
                "score": float("-inf"),
                "candidate_id": -1,
                "conf_percentile": float(conf_percentile),
                "stride": int(stride),
                "frames": ",".join(frame_set),
                "frame_count": len(frame_set),
                "correspondence_count": 0,
                "coarse_threshold": 0.0,
                "inlier_ratio": 0.0,
                "inlier_count": 0,
                "median_error": float("inf"),
                "p90_error": float("inf"),
                "rmse": float("inf"),
                "mean_error": float("inf"),
                "status": f"failed: {exc}",
            }
            all_rows.append(row)
            detailed.append({"run_id": run_id, "status": "failed", "error": str(exc), "config": config})

        print(
            "[{done}/{total}] conf={conf:g} stride={stride} frames={frames}".format(
                done=run_id,
                total=len(args.conf_percentiles) * len(args.strides) * len(frame_sets),
                conf=conf_percentile,
                stride=stride,
                frames=",".join(frame_set),
            ),
            flush=True,
        )

    sorted_rows = sorted(all_rows, key=lambda item: item["score"], reverse=True)
    write_csv(run_dir / "summary.csv", sorted_rows)
    write_markdown(run_dir / "summary.md", sorted_rows, args.markdown_top_n)
    write_json(
        run_dir / "summary.json",
        {
            "run_name": run_name,
            "shared_frames": shared_frames,
            "frame_sets": frame_sets,
            "conf_percentiles": args.conf_percentiles,
            "strides": args.strides,
            "ransac_thresholds": args.ransac_thresholds,
            "ransac_iterations": args.ransac_iterations,
            "rows": sorted_rows,
            "details": detailed,
        },
    )
    return {"run_dir": str(run_dir), "rows": sorted_rows[: args.markdown_top_n]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep shared-frame rigid coarse registration parameters.")
    parser.add_argument("--config", type=Path, default=DEFAULT_ALGORITHM_CONFIG)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--conf-percentiles", type=float, nargs="+", default=[0.0, 5.0, 10.0, 20.0])
    parser.add_argument("--strides", type=int, nargs="+", default=[6, 4, 3, 2])
    parser.add_argument("--ransac-thresholds", type=float, nargs="+", default=[0.18, 0.14, 0.10, 0.06])
    parser.add_argument("--ransac-iterations", type=int, default=300)
    parser.add_argument("--frame-combo-size", type=int, default=3)
    parser.add_argument("--include-all-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--markdown-top-n", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    print(json.dumps(json_safe(run_sweep(parse_args(argv))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class StepMetrics:
    """离线 scale gate 诊断所需的稳定字段。

    这里故意不依赖主 pipeline 的内部对象，只读取 result artifacts。目标是先
    验证 gate 的判别能力，再决定是否把规则迁入 `pcr/`。
    """

    step: int
    submap: str
    source_id: str
    target_id: str
    pose_status: str
    fusion_status: str
    reasons: str
    shared_median_error: float
    shared_p90_error: float
    icp_fitness: float
    icp_rmse: float
    icp_translation_delta: float
    icp_rotation_delta_deg: float
    accepted_ratio: float
    duplicate_ratio: float
    conflict_ratio: float
    accepted_points: int
    duplicate_points: int
    conflict_points: int
    accepted_extent_x: float
    accepted_extent_y: float
    accepted_extent_z: float
    accepted_extent_max: float
    accepted_xz_area: float
    accepted_xz_density: float
    accepted_linear_ratio: float
    accepted_planar_ratio: float
    accepted_thin_ratio: float
    base_icp_suspect: bool
    broad_accepted_region: bool
    scale_shadow_gate_v1: bool


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_points(path: Path) -> np.ndarray:
    if not path.exists():
        return np.empty((0, 3), dtype=float)
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        return np.empty((0, 3), dtype=float)
    return np.asarray(cloud.points, dtype=float)


def accepted_geometry_features(step_dir: Path) -> dict[str, float]:
    points = read_points(step_dir / "fusion_debug" / "accepted_points.ply")
    if len(points) < 3:
        return {
            "accepted_extent_x": 0.0,
            "accepted_extent_y": 0.0,
            "accepted_extent_z": 0.0,
            "accepted_extent_max": 0.0,
            "accepted_xz_area": 0.0,
            "accepted_xz_density": 0.0,
            "accepted_linear_ratio": 0.0,
            "accepted_planar_ratio": 0.0,
            "accepted_thin_ratio": 0.0,
        }

    extent = points.max(axis=0) - points.min(axis=0)
    xz_area = float(max(extent[0], 1e-9) * max(extent[2], 1e-9))
    centered = points - points.mean(axis=0)
    covariance = np.cov(centered.T)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
    largest = float(max(eigenvalues[0], 1e-12))

    # PCA ratios are only descriptive. The current candidate gate relies on
    # extent/area because submap_007 scale shadows are broad, low-support regions.
    linear_ratio = float((eigenvalues[0] - eigenvalues[1]) / largest)
    planar_ratio = float((eigenvalues[1] - eigenvalues[2]) / largest)
    thin_ratio = float(eigenvalues[2] / largest)
    return {
        "accepted_extent_x": float(extent[0]),
        "accepted_extent_y": float(extent[1]),
        "accepted_extent_z": float(extent[2]),
        "accepted_extent_max": float(extent.max()),
        "accepted_xz_area": xz_area,
        "accepted_xz_density": float(len(points) / xz_area),
        "accepted_linear_ratio": linear_ratio,
        "accepted_planar_ratio": planar_ratio,
        "accepted_thin_ratio": thin_ratio,
    }


def discover_step_dirs(run_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in (run_dir / "steps").glob("step_*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("_", 2)[1])
        except (IndexError, ValueError):
            continue
        result[step] = path
    return result


def build_metrics(run_dir: Path) -> list[StepMetrics]:
    summary = load_json(run_dir / "summary.json")
    step_dirs = discover_step_dirs(run_dir)
    rows: list[StepMetrics] = []
    for row in summary["steps"]:
        fusion = row.get("fusion")
        if not fusion:
            continue
        step = int(row["step_index"])
        step_dir = step_dirs[step]
        quality = load_json(step_dir / "quality_report.json")["quality"]
        metrics = quality["metrics"]
        geom = accepted_geometry_features(step_dir)

        base_icp_suspect = (
            float(metrics["icp_fitness"]) < 0.55
            and float(fusion["conflict_ratio"]) > 0.55
            and float(fusion["accepted_ratio"]) > 0.05
        )
        broad_accepted_region = (
            geom["accepted_xz_area"] > 25.0
            and geom["accepted_extent_max"] > 7.0
        )
        scale_shadow_gate_v1 = base_icp_suspect and broad_accepted_region

        rows.append(
            StepMetrics(
                step=step,
                submap=str(row["active_submap_id"]),
                source_id=str(row["source_id"]),
                target_id=str(row["target_id"]),
                pose_status=str(row["pose_status"]),
                fusion_status=str(row["fusion_status"]),
                reasons=", ".join(row.get("reasons", ())),
                shared_median_error=float(metrics["shared_median_error"]),
                shared_p90_error=float(metrics["shared_p90_error"]),
                icp_fitness=float(metrics["icp_fitness"]),
                icp_rmse=float(metrics["icp_rmse"]),
                icp_translation_delta=float(metrics["icp_translation_delta"]),
                icp_rotation_delta_deg=float(metrics["icp_rotation_delta_deg"]),
                accepted_ratio=float(fusion["accepted_ratio"]),
                duplicate_ratio=float(fusion["duplicate_ratio"]),
                conflict_ratio=float(fusion["conflict_ratio"]),
                accepted_points=int(fusion["accepted_points"]),
                duplicate_points=int(fusion["duplicate_points"]),
                conflict_points=int(fusion["conflict_points"]),
                accepted_extent_x=geom["accepted_extent_x"],
                accepted_extent_y=geom["accepted_extent_y"],
                accepted_extent_z=geom["accepted_extent_z"],
                accepted_extent_max=geom["accepted_extent_max"],
                accepted_xz_area=geom["accepted_xz_area"],
                accepted_xz_density=geom["accepted_xz_density"],
                accepted_linear_ratio=geom["accepted_linear_ratio"],
                accepted_planar_ratio=geom["accepted_planar_ratio"],
                accepted_thin_ratio=geom["accepted_thin_ratio"],
                base_icp_suspect=base_icp_suspect,
                broad_accepted_region=broad_accepted_region,
                scale_shadow_gate_v1=scale_shadow_gate_v1,
            )
        )
    return rows


def write_csv(path: Path, rows: list[StepMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(StepMetrics.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def fmt_steps(rows: list[StepMetrics]) -> str:
    return ", ".join(str(item.step) for item in rows) or "none"


def markdown_table(rows: list[StepMetrics], *, limit: int | None = None) -> list[str]:
    selected = rows if limit is None else rows[:limit]
    lines = [
        "| step | submap | pose | fusion | p90 | fitness | accR | dupR | confR | xz_area | extent | gate |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in selected:
        lines.append(
            "| {step} | {submap} | {pose} | {fusion} | {p90:.3f} | {fit:.3f} | {acc:.3f} | {dup:.3f} | {conf:.3f} | {area:.1f} | {extent:.1f} | {gate} |".format(
                step=row.step,
                submap=row.submap,
                pose=row.pose_status,
                fusion=row.fusion_status,
                p90=row.shared_p90_error,
                fit=row.icp_fitness,
                acc=row.accepted_ratio,
                dup=row.duplicate_ratio,
                conf=row.conflict_ratio,
                area=row.accepted_xz_area,
                extent=row.accepted_extent_max,
                gate="yes" if row.scale_shadow_gate_v1 else "",
            )
        )
    return lines


def write_report(path: Path, rows: list[StepMetrics], run_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hits = [row for row in rows if row.scale_shadow_gate_v1]
    pre20 = [row for row in rows if row.step <= 20]
    pre20_hits = [row for row in pre20 if row.scale_shadow_gate_v1]
    submap7 = [row for row in rows if row.submap == "submap_007"]
    submap7_hits = [row for row in submap7 if row.scale_shadow_gate_v1]
    step71_80 = [row for row in rows if 71 <= row.step <= 80]
    step71_80_hits = [row for row in step71_80 if row.scale_shadow_gate_v1]

    lines = [
        "# Scale Gate Analysis",
        "",
        f"Run: `{run_dir}`",
        "",
        "## Candidate Gate",
        "",
        "```text",
        "base_icp_suspect = icp_fitness < 0.55 and conflict_ratio > 0.55 and accepted_ratio > 0.05",
        "broad_accepted_region = accepted_xz_area > 25.0 and accepted_extent_max > 7.0",
        "scale_shadow_gate_v1 = base_icp_suspect and broad_accepted_region",
        "```",
        "",
        "Rationale: the broad-region term separates submap_007 scale-shadow steps from early steps that also have low ICP fitness and high conflict, but whose accepted points are compact local additions.",
        "",
        "## Hit Summary",
        "",
        f"- Fusion attempts analyzed: {len(rows)}",
        f"- Gate hits: {len(hits)}",
        f"- First 20 hits: {len(pre20_hits)} ({fmt_steps(pre20_hits)})",
        f"- Step 71-80 hits: {len(step71_80_hits)} ({fmt_steps(step71_80_hits)})",
        f"- Submap_007 hits: {len(submap7_hits)} ({fmt_steps(submap7_hits)})",
        "",
        "## First 20 Impact",
        "",
        *markdown_table(pre20),
        "",
        "## Step 71-80",
        "",
        *markdown_table(step71_80),
        "",
        "## All Gate Hits",
        "",
        *markdown_table(hits, limit=80),
        "",
        "## Notes",
        "",
        "- This is an offline diagnostic prototype. It does not modify the main mapping pipeline.",
        "- `accepted_xz_area` is computed from `fusion_debug/accepted_points.ply`, so it reflects the points that the current fusion logic would write as new structure.",
        "- The run does not save a per-step pre-fusion world. This prototype therefore uses accepted cloud morphology rather than nearest-surface features against the exact world-before-step.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate scale-shadow fusion gates from submap artifacts.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("result/submap_walkforward/submap_327_align_only_se3"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("playground"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_metrics(args.run_dir)
    write_csv(args.output_dir / "scale_gate_metrics.csv", rows)
    write_report(args.output_dir / "scale_gate_report.md", rows, args.run_dir)
    hits = [row for row in rows if row.scale_shadow_gate_v1]
    pre20_hits = [row for row in rows if row.step <= 20 and row.scale_shadow_gate_v1]
    submap7_hits = [row for row in rows if row.submap == "submap_007" and row.scale_shadow_gate_v1]
    print(f"analyzed={len(rows)} hits={len(hits)} pre20_hits={fmt_steps(pre20_hits)} submap7_hits={fmt_steps(submap7_hits)}")
    print(f"wrote {args.output_dir / 'scale_gate_metrics.csv'}")
    print(f"wrote {args.output_dir / 'scale_gate_report.md'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PolicyConfig:
    """离线 submap 策略试验参数。

    这里不修改主 pipeline，只用已有 artifact 评估“如果 submap 切换按质量驱动，
    哪些地方应该继续持有当前 submap，哪些地方应该标记 tracking weak/lost”。
    """

    min_steps: int = 8
    target_steps: int = 10
    max_steps: int = 20
    min_fused_for_rotation: int = 5
    recent_window: int = 6
    min_recent_fused: int = 2
    max_steps_since_fused: int = 6
    max_provisional_chain: int = 2


@dataclass(frozen=True)
class StepRecord:
    step: int
    actual_submap: str
    source_id: str
    target_id: str
    pose_status: str
    fusion_status: str
    reasons: tuple[str, ...]
    committed_fusion: bool
    scale_quarantined: bool
    target_transform_quality: str
    target_provisional_chain: int
    shared_p90_error: float
    icp_fitness: float
    icp_translation_delta: float
    icp_rotation_delta_deg: float


@dataclass(frozen=True)
class SegmentRecord:
    simulated_submap: str
    start_step: int
    end_step: int
    steps: int
    fused_steps: int
    pose_rejected_steps: int
    scale_quarantined_steps: int
    rotation_decision: str
    reason: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def step_dir(run_dir: Path, step: int) -> Path:
    matches = sorted((run_dir / "steps").glob(f"step_{step:03d}_*"))
    if not matches:
        raise FileNotFoundError(f"No artifact directory for step {step}")
    return matches[0]


def quality_metrics(run_dir: Path, step: int) -> dict[str, Any]:
    report = load_json(step_dir(run_dir, step) / "quality_report.json")
    return report["quality"]["metrics"]


def fusion_committed(row: dict[str, Any]) -> bool:
    fusion = row.get("fusion")
    if not fusion:
        return False
    if row["fusion_status"] == "rejected":
        return False
    return bool(fusion.get("params", {}).get("committed", True))


def build_step_records(run_dir: Path) -> list[StepRecord]:
    summary = load_json(run_dir / "summary.json")
    transform_quality: dict[str, str] = {}
    provisional_chain: dict[str, int] = {}
    baseline_id = summary["steps"][0]["target_id"]
    transform_quality[baseline_id] = "fused"
    provisional_chain[baseline_id] = 0
    records: list[StepRecord] = []

    for row in summary["steps"]:
        step = int(row["step_index"])
        metrics = quality_metrics(run_dir, step)
        target_id = str(row["target_id"])
        target_quality = transform_quality.get(target_id, "missing")
        target_chain = provisional_chain.get(target_id, 0)
        reasons = tuple(str(item) for item in row.get("reasons", ()))
        committed = fusion_committed(row)
        scale_quarantined = "fusion_scale_shadow_quarantined" in reasons

        records.append(
            StepRecord(
                step=step,
                actual_submap=str(row["active_submap_id"]),
                source_id=str(row["source_id"]),
                target_id=target_id,
                pose_status=str(row["pose_status"]),
                fusion_status=str(row["fusion_status"]),
                reasons=reasons,
                committed_fusion=committed,
                scale_quarantined=scale_quarantined,
                target_transform_quality=target_quality,
                target_provisional_chain=target_chain,
                shared_p90_error=float(metrics.get("shared_p90_error", 0.0)),
                icp_fitness=float(metrics.get("icp_fitness", 0.0)),
                icp_translation_delta=float(metrics.get("icp_translation_delta", 0.0)),
                icp_rotation_delta_deg=float(metrics.get("icp_rotation_delta_deg", 0.0)),
            )
        )

        source_id = str(row["source_id"])
        if row["pose_status"] == "rejected":
            transform_quality[source_id] = "provisional"
            provisional_chain[source_id] = target_chain + 1
        elif committed:
            transform_quality[source_id] = "fused"
            provisional_chain[source_id] = 0
        else:
            transform_quality[source_id] = "registered"
            provisional_chain[source_id] = 0

    return records


def simulate_segments(records: list[StepRecord], config: PolicyConfig) -> list[SegmentRecord]:
    segments: list[SegmentRecord] = []
    current_id = 0
    start = records[0].step
    step_count = 0
    fused_count = 0
    pose_rejected_count = 0
    scale_count = 0
    recent_fused: list[bool] = []
    steps_since_fused = 0

    def close_segment(end_step: int, decision: str, reason: str) -> None:
        nonlocal current_id, start, step_count, fused_count, pose_rejected_count, scale_count, recent_fused, steps_since_fused
        segments.append(
            SegmentRecord(
                simulated_submap=f"sim_submap_{current_id:03d}",
                start_step=start,
                end_step=end_step,
                steps=step_count,
                fused_steps=fused_count,
                pose_rejected_steps=pose_rejected_count,
                scale_quarantined_steps=scale_count,
                rotation_decision=decision,
                reason=reason,
            )
        )
        current_id += 1
        start = end_step + 1
        step_count = 0
        fused_count = 0
        pose_rejected_count = 0
        scale_count = 0
        recent_fused = []
        steps_since_fused = 0

    for record in records:
        step_count += 1
        if record.committed_fusion:
            fused_count += 1
            steps_since_fused = 0
        else:
            steps_since_fused += 1
        if record.pose_status == "rejected":
            pose_rejected_count += 1
        if record.scale_quarantined:
            scale_count += 1
        recent_fused.append(record.committed_fusion)
        recent_fused = recent_fused[-config.recent_window :]
        recent_count = sum(recent_fused)

        can_rotate = (
            step_count >= config.min_steps
            and fused_count >= config.min_fused_for_rotation
            and recent_count >= config.min_recent_fused
            and steps_since_fused <= config.max_steps_since_fused
        )
        if step_count >= config.target_steps and can_rotate:
            close_segment(record.step, "trusted_rotate", "quality_window_satisfied")
        elif step_count >= config.max_steps:
            if fused_count >= config.min_fused_for_rotation and recent_count >= 1:
                close_segment(record.step, "weak_rotate", "max_steps_reached_with_some_recent_fusion")
            else:
                close_segment(record.step, "detached_or_hold", "max_steps_reached_without_reliable_fusion")

    if step_count:
        close_segment(records[-1].step, "final_open", "end_of_sequence")
    return segments


def edge_jumps(run_dir: Path) -> list[dict[str, Any]]:
    chain = load_json(run_dir / "submap_chain.json")
    result = []
    for edge in chain.get("edges", ()):
        matrix = np.asarray(edge["transform"]["matrix"], dtype=float)
        translation_norm = float(np.linalg.norm(matrix[:3, 3]))
        yaw_like_deg = float(math.degrees(math.atan2(matrix[0, 2], matrix[0, 0])))
        suspicious = translation_norm > 3.0 or abs(yaw_like_deg) > 90.0
        result.append(
            {
                "source": edge["source_submap_id"],
                "target": edge["target_submap_id"],
                "reason": edge.get("reason", ""),
                "translation_norm": translation_norm,
                "yaw_like_deg": yaw_like_deg,
                "suspicious": suspicious,
            }
        )
    return result


def write_steps_csv(path: Path, records: list[StepRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(StepRecord.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row = record.__dict__.copy()
            row["reasons"] = ";".join(record.reasons)
            writer.writerow(row)


def write_segments_csv(path: Path, segments: list[SegmentRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(SegmentRecord.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for segment in segments:
            writer.writerow(segment.__dict__)


def write_report(
    path: Path,
    *,
    run_dir: Path,
    records: list[StepRecord],
    segments: list[SegmentRecord],
    jumps: list[dict[str, Any]],
    config: PolicyConfig,
) -> None:
    pose_rejected = sum(item.pose_status == "rejected" for item in records)
    committed = sum(item.committed_fusion for item in records)
    quarantined = sum(item.scale_quarantined for item in records)
    target_provisional = [item for item in records if item.target_transform_quality == "provisional"]
    long_provisional = [item for item in target_provisional if item.target_provisional_chain > config.max_provisional_chain]
    trusted = [item for item in segments if item.rotation_decision == "trusted_rotate"]
    weak = [item for item in segments if item.rotation_decision == "weak_rotate"]
    detached = [item for item in segments if item.rotation_decision == "detached_or_hold"]
    suspicious_jumps = [item for item in jumps if item["suspicious"]]

    lines = [
        "# Submap Policy Simulation",
        "",
        f"Run: `{run_dir}`",
        "",
        "## Policy",
        "",
        "```text",
        f"min_steps = {config.min_steps}",
        f"target_steps = {config.target_steps}",
        f"max_steps = {config.max_steps}",
        f"min_fused_for_rotation = {config.min_fused_for_rotation}",
        f"recent_window = {config.recent_window}",
        f"min_recent_fused = {config.min_recent_fused}",
        f"max_steps_since_fused = {config.max_steps_since_fused}",
        f"max_provisional_chain = {config.max_provisional_chain}",
        "```",
        "",
        "## Summary",
        "",
        f"- Steps: {len(records)}",
        f"- Pose rejected steps: {pose_rejected}",
        f"- Committed fusion steps: {committed}",
        f"- Scale quarantined steps: {quarantined}",
        f"- Steps using provisional target transform: {len(target_provisional)}",
        f"- Steps with provisional chain > {config.max_provisional_chain}: {len(long_provisional)}",
        f"- Simulated segments: {len(segments)}",
        f"- Trusted rotations: {len(trusted)}",
        f"- Weak rotations: {len(weak)}",
        f"- Detached/hold segments: {len(detached)}",
        f"- Suspicious existing submap edges: {len(suspicious_jumps)}",
        "",
        "## Detached Or Hold Segments",
        "",
        "| sim submap | steps | fused | pose rejected | scale gate | decision | reason |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for segment in detached:
        lines.append(
            f"| {segment.simulated_submap} | {segment.start_step}-{segment.end_step} | {segment.fused_steps} | {segment.pose_rejected_steps} | {segment.scale_quarantined_steps} | {segment.rotation_decision} | {segment.reason} |"
        )
    if not detached:
        lines.append("| none |  |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Existing Suspicious Submap Edges",
            "",
            "| edge | anchor | translation m | yaw-like deg |",
            "|---|---|---:|---:|",
        ]
    )
    for item in suspicious_jumps:
        lines.append(
            f"| {item['source']} -> {item['target']} | {item['reason']} | {item['translation_norm']:.3f} | {item['yaw_like_deg']:.1f} |"
        )
    if not suspicious_jumps:
        lines.append("| none |  |  |  |")

    lines.extend(
        [
            "",
            "## Long Provisional Target Chain",
            "",
            "| step | target | chain | pose | fusion | reasons |",
            "|---:|---|---:|---|---|---|",
        ]
    )
    for item in long_provisional[:80]:
        lines.append(
            f"| {item.step} | {item.target_id} | {item.target_provisional_chain} | {item.pose_status} | {item.fusion_status} | {'; '.join(item.reasons)} |"
        )
    if not long_provisional:
        lines.append("| none |  |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A detached/hold segment means fixed-size submap rotation would be unsafe because the local map did not grow enough.",
            "- A long provisional target chain means ICP initialization is being propagated through rejected poses rather than through fused map anchors.",
            "- Suspicious existing edges are not proof of visual failure, but they are high-priority candidates for submap edge sanity gates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline simulation for quality-driven submap policy.")
    parser.add_argument("--run-dir", type=Path, default=Path("result/submap_walkforward/submap_327_align_only_se3_scale_gate_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("playground"))
    parser.add_argument("--min-steps", type=int, default=8)
    parser.add_argument("--target-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--min-fused-for-rotation", type=int, default=5)
    parser.add_argument("--recent-window", type=int, default=6)
    parser.add_argument("--min-recent-fused", type=int, default=2)
    parser.add_argument("--max-steps-since-fused", type=int, default=6)
    parser.add_argument("--max-provisional-chain", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PolicyConfig(
        min_steps=args.min_steps,
        target_steps=args.target_steps,
        max_steps=args.max_steps,
        min_fused_for_rotation=args.min_fused_for_rotation,
        recent_window=args.recent_window,
        min_recent_fused=args.min_recent_fused,
        max_steps_since_fused=args.max_steps_since_fused,
        max_provisional_chain=args.max_provisional_chain,
    )
    records = build_step_records(args.run_dir)
    segments = simulate_segments(records, config)
    jumps = edge_jumps(args.run_dir)
    write_steps_csv(args.output_dir / "submap_policy_steps.csv", records)
    write_segments_csv(args.output_dir / "submap_policy_segments.csv", segments)
    write_report(
        args.output_dir / "submap_policy_report.md",
        run_dir=args.run_dir,
        records=records,
        segments=segments,
        jumps=jumps,
        config=config,
    )
    detached = [item for item in segments if item.rotation_decision == "detached_or_hold"]
    trusted = [item for item in segments if item.rotation_decision == "trusted_rotate"]
    suspicious = [item for item in jumps if item["suspicious"]]
    print(f"steps={len(records)} segments={len(segments)} trusted={len(trusted)} detached_or_hold={len(detached)} suspicious_edges={len(suspicious)}")
    print(f"wrote {args.output_dir / 'submap_policy_report.md'}")


if __name__ == "__main__":
    main()

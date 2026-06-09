from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from utils.pointcloud import clone_transform, make_overlay, write_point_cloud
from utils.types import Candidate, RegistrationResult


def json_safe(value: Any) -> Any:
    """把 numpy/path 等对象转换成 JSON/YAML 可写类型。"""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    return value


def write_json(path: str | Path, payload: dict) -> None:
    """写 JSON 文件，保留中文。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)


def write_yaml(path: str | Path, payload: dict) -> None:
    """写 YAML 文件，用于保存 run_config 快照。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(json_safe(payload), handle, sort_keys=False, allow_unicode=True)


def write_matrix(path: str | Path, matrix: np.ndarray) -> None:
    """以文本形式保存 4x4 变换矩阵。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.10f")


def candidate_to_dict(candidate: Candidate) -> dict:
    """把 Candidate 转成可序列化字典。"""

    return {
        "candidate_id": candidate.candidate_id,
        "coarse_score": candidate.coarse_score,
        "cluster_id": candidate.cluster_id,
        "cluster_size": candidate.cluster_size,
        "cluster_top_score_count": candidate.cluster_top_score_count,
        "scale_values": candidate.scale_values,
        "vote_peak_ratio": candidate.vote_peak_ratio,
        "metrics": candidate.metrics,
        "rank_score": candidate.rank_score,
        "metadata": candidate.metadata,
        "matrix": candidate.matrix,
    }


def write_top_candidates(path: str | Path, candidates: list[Candidate]) -> None:
    """写 top candidates Markdown 表，便于人工比较候选。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| rank | candidate | cluster | score | fitness | trimmed | scale | status_hint |",
        "|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for rank, candidate in enumerate(candidates, 1):
        metrics = candidate.metrics
        scale = candidate.scale_values or {}
        scale_text = ",".join(f"{k}={v:.3f}" for k, v in scale.items())
        lines.append(
            "| {rank} | {cid} | {cluster} | {score:.6f} | {fitness:.6f} | {trimmed} | {scale} | {hint} |".format(
                rank=rank,
                cid=candidate.candidate_id,
                cluster=candidate.cluster_id,
                score=candidate.rank_score,
                fitness=float(metrics.get("eval_fitness") or 0.0),
                trimmed="None"
                if metrics.get("eval_trimmed_mean_nn_dist") is None
                else f"{float(metrics['eval_trimmed_mean_nn_dist']):.6f}",
                scale=scale_text,
                hint=candidate.metadata.get("status_hint", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_pair_outputs(
    output_dir: str | Path,
    result: RegistrationResult,
    source_raw,
    target_raw,
) -> None:
    """保存一次 pairwise 注册的标准产物。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registered = clone_transform(source_raw, result.matrix)
    overlay = make_overlay(target_raw, registered)

    write_matrix(output_dir / "matrix.txt", result.matrix)
    write_json(
        output_dir / "metrics.json",
        {
            "algorithm": result.algorithm,
            "source": result.source_path,
            "target": result.target_path,
            "status": result.status,
            "metrics": result.metrics,
            "top_candidates": [candidate_to_dict(c) for c in result.top_candidates],
        },
    )
    write_point_cloud(output_dir / "registered_source.ply", registered)
    write_point_cloud(output_dir / "overlay_top1.ply", overlay)
    write_top_candidates(output_dir / "top_candidates.md", result.top_candidates)

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from utils import io, reporting


EXPERIMENT_NAME = "candidate_ranker"

TASKS = [
    {
        "task_id": "subdir_incremental16_to_world14",
        "run_dir": "results/specified/multi_comb/ransac_all_candidate_bank/ransac_all_candidate_bank_20260604_133558",
        "positive_candidate_ids": [49, 191, 86, 122, 7, 168],
    },
    {
        "task_id": "root_incremental16_to_world14",
        "run_dir": "results/specified/multi_comb/ransac_all_candidate_bank/ransac_all_candidate_bank_20260604_151518",
        "positive_candidate_ids": [33, 55, 61, 70, 76, 80, 81, 95, 110, 115, 116, 130, 138, 184],
    },
]

FEATURES = [
    "eval_fitness",
    "eval_inlier_rmse",
    "eval_trimmed_mean_nn_dist",
    "eval_overlap_ratio",
    "eval_translation_norm",
    "coverage_score",
    "plane_degeneracy",
    "inlier_count",
    "algorithm_fitness",
    "algorithm_inlier_rmse",
    "algorithm_correspondence_set_size",
    "score",
    "rotation_cluster_size",
    "rotation_cluster_rank_by_size",
    "rotation_cluster_best_score",
    "rotation_cluster_mean_score",
    "rotation_cluster_best_fitness",
    "rotation_cluster_mean_trimmed",
    "rotation_cluster_mean_plane_degeneracy",
    "rotation_cluster_mean_coverage",
]


def main() -> int:
    rows = _load_training_rows(TASKS)
    run_dir = reporting.make_run_dir(Path("results") / "specified" / "multi_comb" / EXPERIMENT_NAME, EXPERIMENT_NAME)
    _write_table(run_dir / "training_data.csv", rows)
    _write_json(run_dir / "training_data.json", rows)

    evaluations = []
    ranked_by_task: dict[str, list[dict[str, Any]]] = {}
    for holdout in [task["task_id"] for task in TASKS]:
        train_rows = [row for row in rows if row["task_id"] != holdout]
        test_rows = [row for row in rows if row["task_id"] == holdout]
        model = _fit_logistic(train_rows)
        ranked = _rank_rows(test_rows, model)
        ranked_by_task[holdout] = ranked
        evaluations.append(_evaluate_task(holdout, ranked, score_key="logistic_score"))
        _write_table(run_dir / f"ranked_candidates_{holdout}.csv", ranked)

    full_model = _fit_logistic(rows)
    _write_json(run_dir / "model_coefficients.json", _model_payload(full_model))
    _write_json(run_dir / "evaluation.json", evaluations)
    _write_summary(run_dir / "summary.md", rows, evaluations, full_model, ranked_by_task)
    _write_summary(run_dir / "report" / "summary.md", rows, evaluations, full_model, ranked_by_task)
    _write_visualization_commands(run_dir / "report" / "visualization_commands.md", ranked_by_task)
    print(f"wrote candidate ranker results to {run_dir}")
    return 0


def _load_training_rows(tasks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        positive_ids = {int(v) for v in task["positive_candidate_ids"]}
        run_dir = Path(str(task["run_dir"]))
        records = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        records = [record for record in records if record.get("status") == "success" and record.get("matrix_path")]
        clusters = _rotation_clusters(records)
        cluster_by_candidate = {int(item["candidate_id"]): cluster for cluster in clusters for item in cluster["items"]}
        cluster_rank = {cluster["cluster_id"]: rank for rank, cluster in enumerate(sorted(clusters, key=lambda c: len(c["items"]), reverse=True), start=1)}
        for record in records:
            candidate_id = int(record["candidate_id"])
            cluster = cluster_by_candidate[candidate_id]
            row = {k: _json_value(v) for k, v in record.items() if not str(k).startswith("_")}
            row.update(_cluster_features(cluster, cluster_rank[cluster["cluster_id"]]))
            row["task_id"] = task_id
            row["run_dir"] = str(run_dir)
            row["candidate_id"] = candidate_id
            row["is_positive"] = 1 if candidate_id in positive_ids else 0
            row["cluster_positive_count"] = sum(1 for item in cluster["items"] if int(item["candidate_id"]) in positive_ids)
            row["cluster_is_positive"] = 1 if row["cluster_positive_count"] else 0
            output.append(row)
    return output


def _rotation_clusters(records: list[dict[str, Any]], threshold_deg: float = 5.0) -> list[dict[str, Any]]:
    matrices = {int(record["candidate_id"]): io.read_matrix(record["matrix_path"]) for record in records}
    clusters: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: int(r["candidate_id"])):
        matrix = matrices[int(record["candidate_id"])]
        placed = False
        for cluster in clusters:
            if _rotation_diff_deg(matrix, cluster["representative_matrix"]) <= threshold_deg:
                cluster["items"].append(record)
                placed = True
                break
        if not placed:
            clusters.append({"cluster_id": len(clusters) + 1, "representative_matrix": matrix, "items": [record]})
    return clusters


def _cluster_features(cluster: Mapping[str, Any], rank_by_size: int) -> dict[str, Any]:
    items = list(cluster["items"])
    scores = [_float(item.get("score"), 0.0) for item in items]
    fitness = [_float(item.get("eval_fitness"), 0.0) for item in items]
    trimmed = [_float(item.get("eval_trimmed_mean_nn_dist"), 0.0) for item in items]
    plane = [_float(item.get("plane_degeneracy"), 0.0) for item in items]
    coverage = [_float(item.get("coverage_score"), 0.0) for item in items]
    return {
        "rotation_cluster_id": int(cluster["cluster_id"]),
        "rotation_cluster_size": len(items),
        "rotation_cluster_rank_by_size": rank_by_size,
        "rotation_cluster_best_score": max(scores),
        "rotation_cluster_mean_score": float(np.mean(scores)),
        "rotation_cluster_best_fitness": max(fitness),
        "rotation_cluster_mean_trimmed": float(np.mean(trimmed)),
        "rotation_cluster_mean_plane_degeneracy": float(np.mean(plane)),
        "rotation_cluster_mean_coverage": float(np.mean(coverage)),
    }


def _fit_logistic(rows: list[dict[str, Any]]) -> dict[str, Any]:
    x = _feature_matrix(rows)
    y = np.asarray([int(row["is_positive"]) for row in rows], dtype=float)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1.0e-9] = 1.0
    z = (x - mean) / std
    z = np.hstack([np.ones((len(z), 1)), z])
    weights = np.zeros(z.shape[1], dtype=float)
    positives = max(float(np.sum(y)), 1.0)
    negatives = max(float(len(y) - np.sum(y)), 1.0)
    sample_weights = np.where(y > 0.5, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    lr = 0.05
    l2 = 0.05
    for _ in range(5000):
        pred = _sigmoid(z @ weights)
        grad = (z.T @ ((pred - y) * sample_weights)) / len(y)
        grad[1:] += l2 * weights[1:] / len(y)
        weights -= lr * grad
    return {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "weights": weights.tolist()}


def _rank_rows(rows: list[dict[str, Any]], model: Mapping[str, Any]) -> list[dict[str, Any]]:
    x = _feature_matrix(rows)
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    z = (x - mean) / std
    z = np.hstack([np.ones((len(z), 1)), z])
    scores = _sigmoid(z @ weights)
    ranked = []
    for row, score in zip(rows, scores):
        updated = dict(row)
        updated["logistic_score"] = float(score)
        ranked.append(updated)
    ranked.sort(key=lambda r: float(r["logistic_score"]), reverse=True)
    for idx, row in enumerate(ranked, start=1):
        row["logistic_rank"] = idx
    return ranked


def _evaluate_task(task_id: str, ranked: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    cluster_scores: dict[int, dict[str, Any]] = {}
    for row in ranked:
        cluster_id = int(row["rotation_cluster_id"])
        entry = cluster_scores.setdefault(
            cluster_id,
            {
                "rotation_cluster_id": cluster_id,
                "score": float("-inf"),
                "size": int(row["rotation_cluster_size"]),
                "positive": bool(row["cluster_is_positive"]),
                "best_candidate_id": None,
            },
        )
        score = float(row[score_key])
        if score > float(entry["score"]):
            entry["score"] = score
            entry["best_candidate_id"] = int(row["candidate_id"])
    ranked_clusters = sorted(cluster_scores.values(), key=lambda c: float(c["score"]), reverse=True)
    positive_cluster_ranks = [idx for idx, cluster in enumerate(ranked_clusters, start=1) if cluster["positive"]]
    positive_candidate_ranks = [int(row["logistic_rank"]) for row in ranked if row["is_positive"]]
    return {
        "task_id": task_id,
        "positive_cluster_ranks": positive_cluster_ranks,
        "best_positive_cluster_rank": min(positive_cluster_ranks) if positive_cluster_ranks else None,
        "top5_cluster_contains_positive": bool(positive_cluster_ranks and min(positive_cluster_ranks) <= 5),
        "positive_candidate_ranks": positive_candidate_ranks,
        "best_positive_candidate_rank": min(positive_candidate_ranks) if positive_candidate_ranks else None,
        "top10_candidate_contains_positive": bool(positive_candidate_ranks and min(positive_candidate_ranks) <= 10),
        "top_clusters": ranked_clusters[:10],
    }


def _feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([[_float(row.get(name), 0.0) for name in FEATURES] for row in rows], dtype=float)


def _model_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    weights = list(model["weights"])
    feature_weights = [{"feature": feature, "weight": float(weight)} for feature, weight in zip(FEATURES, weights[1:])]
    feature_weights.sort(key=lambda item: abs(item["weight"]), reverse=True)
    return {
        "intercept": float(weights[0]),
        "feature_weights": feature_weights,
        "mean": model["mean"],
        "std": model["std"],
    }


def _write_summary(path: Path, rows: list[dict[str, Any]], evaluations: list[dict[str, Any]], model: Mapping[str, Any], ranked_by_task: Mapping[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    positives = sum(int(row["is_positive"]) for row in rows)
    lines = [
        "# Candidate Ranker",
        "",
        f"- samples: `{len(rows)}`",
        f"- positives: `{positives}`",
        f"- negatives: `{len(rows) - positives}`",
        f"- model: `weighted logistic regression`",
        f"- validation: `leave-one-task-out`",
        "",
        "## Evaluation",
        "",
        "| task | best positive cluster rank | top5 cluster hit | best positive candidate rank | top10 candidate hit |",
        "|---|---:|---|---:|---|",
    ]
    for item in evaluations:
        lines.append(
            "| {task} | {cluster_rank} | {cluster_hit} | {candidate_rank} | {candidate_hit} |".format(
                task=item["task_id"],
                cluster_rank=item["best_positive_cluster_rank"],
                cluster_hit=item["top5_cluster_contains_positive"],
                candidate_rank=item["best_positive_candidate_rank"],
                candidate_hit=item["top10_candidate_contains_positive"],
            )
        )
    lines.extend(["", "## Strongest Weights", ""])
    for item in _model_payload(model)["feature_weights"][:12]:
        lines.append(f"- `{item['feature']}`: `{item['weight']:.6f}`")
    lines.extend(["", "## Top Logistic Candidates", ""])
    for task_id, ranked in ranked_by_task.items():
        lines.extend([f"### {task_id}", "", "| rank | candidate | cluster | positive | score | matrix |", "|---:|---:|---:|---:|---:|---|"])
        for row in ranked[:10]:
            lines.append(
                "| {rank} | {candidate} | {cluster} | {positive} | {score:.6f} | {matrix} |".format(
                    rank=row["logistic_rank"],
                    candidate=row["candidate_id"],
                    cluster=row["rotation_cluster_id"],
                    positive=row["is_positive"],
                    score=row["logistic_score"],
                    matrix=row["matrix_path"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_visualization_commands(path: Path, ranked_by_task: Mapping[str, list[dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Candidate Ranker Visualization Commands", ""]
    for task_id, ranked in ranked_by_task.items():
        lines.extend([f"## {task_id}", ""])
        rows = _top_diverse_rows(ranked, limit=8)
        for row in rows:
            lines.extend(
                [
                    f"### rank {row['logistic_rank']} candidate {row['candidate_id']}",
                    "",
                    f"- cluster: `{row['rotation_cluster_id']}`",
                    f"- positive: `{row['is_positive']}`",
                    f"- logistic_score: `{row['logistic_score']:.6f}`",
                    "",
                    "```bat",
                    ".env\\python.exe scripts\\visualize_overlay.py --source \"{source}\" --target \"{target}\" --matrix \"{matrix}\"".format(
                        source=_source_for_task(task_id),
                        target=_target_for_task(task_id),
                        matrix=row["matrix_path"],
                    ),
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def _top_diverse_rows(ranked: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = []
    seen_clusters = set()
    for row in ranked:
        cluster_id = int(row["rotation_cluster_id"])
        if cluster_id in seen_clusters:
            continue
        selected.append(row)
        seen_clusters.add(cluster_id)
        if len(selected) >= limit:
            break
    return selected


def _source_for_task(task_id: str) -> str:
    if task_id == "subdir_incremental16_to_world14":
        return "data/raw/16/incremental.ply"
    return "data/raw/16_incremental.ply"


def _target_for_task(task_id: str) -> str:
    if task_id == "subdir_incremental16_to_world14":
        return "data/raw/14/world.ply"
    return "data/raw/14_world.ply"


def _rotation_diff_deg(a: np.ndarray, b: np.ndarray) -> float:
    rotation = a[:3, :3] @ b[:3, :3].T
    value = (float(np.trace(rotation)) - 1.0) / 2.0
    value = min(1.0, max(-1.0, value))
    return float(np.degrees(np.arccos(value)))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_table(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    raise SystemExit(main())

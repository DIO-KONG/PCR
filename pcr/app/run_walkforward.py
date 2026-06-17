from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pcr.artifacts.store import ArtifactStore, json_safe
from pcr.config.loader import build_sequence_task, load_yaml
from pcr.domain import Transform
from pcr.pipeline.sequence import SequencePipeline


DEFAULT_CONFIG = Path("testbench/configs/experiments/shared_frame_walkforward.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 walk-forward CLI 参数。"""

    parser = argparse.ArgumentParser(description="Run shared-frame dynamic-top3 walk-forward registration.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """执行完整 walk-forward。

    app 层只做顶层控制：配置、run 目录、pipeline 调用和 artifact 写入。
    算法细节、world 更新和文件格式分别由更低层模块负责。
    """

    config = load_yaml(args.config)
    run_name = args.run_name or datetime.now().strftime("walkforward_%Y%m%d_%H%M%S")
    result_root = Path(config.get("result_root", "result/walkforward_shared_frame"))
    run_dir = result_root / run_name

    artifact_store = ArtifactStore(run_dir)
    artifact_store.initialize(config=config, overwrite=bool(args.overwrite))
    artifact_store.write_algorithm_record()

    sequence_task = build_sequence_task(config)
    baseline_transform = Transform(source=sequence_task.baseline.batch_id, target="global", matrix=np.eye(4))
    artifact_store.write_initial_transform(baseline_transform)

    pipeline = SequencePipeline()
    output = pipeline.run(task=sequence_task, run_name=run_name, run_dir=run_dir)

    summary_rows: list[dict[str, Any]] = []
    for step_output in output.steps:
        artifact_store.write_step_with_world(
            step_output.update,
            world_before_step=step_output.world_before_step,
            frame_combo_rows=step_output.frame_combo_rows,
        )
        step = step_output.update.step_result
        summary_rows.append(
            {
                "step_index": step.task.step_index,
                "source_id": step.task.source.batch_id,
                "target_id": step.task.target.batch_id,
                "selected_frames": list(step.coarse.frame_selection.selected_frames),
                "coarse_metrics": step.coarse_shared_metrics.to_dict(),
                "icp_metrics": step.refinement.metrics.to_dict(),
                "final_source": step.final_source,
                "fused_world_points": step_output.update.fused_world_points,
                "metrics_path": str(
                    run_dir
                    / f"step_{step.task.step_index:02d}_{step.task.source.batch_id}_to_{step.task.target.batch_id}"
                    / "metrics.json"
                ),
            }
        )

    artifact_store.write_summary(
        output.result,
        summary_rows,
        output.world.transforms_to_global,
    )
    return {
        "run_dir": str(run_dir),
        "final_world_points": output.result.final_world_points,
        "steps": summary_rows,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI main。"""

    print(json.dumps(json_safe(run(parse_args(argv))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

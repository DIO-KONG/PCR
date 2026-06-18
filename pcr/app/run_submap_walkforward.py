from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pcr.artifacts.store import json_safe
from pcr.artifacts.submap_store import SubmapArtifactStore
from pcr.config.loader import build_sequence_task, load_yaml
from pcr.pipeline.submap_sequence import SubmapSequencePipeline


DEFAULT_CONFIG = Path("testbench/configs/experiments/submap_walkforward_327.yaml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 online submap walk-forward CLI 参数。"""

    parser = argparse.ArgumentParser(description="Run online submap shared-frame walk-forward mapping.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """执行 online submap walk-forward。"""

    config = load_yaml(args.config)
    run_name = args.run_name or datetime.now().strftime("submap_walkforward_%Y%m%d_%H%M%S")
    run_dir = Path(config.get("result_root", "result/submap_walkforward")) / run_name

    store = SubmapArtifactStore(run_dir)
    store.initialize(config=config, overwrite=bool(args.overwrite))

    task = build_sequence_task(config)
    output = SubmapSequencePipeline().run(
        task=task,
        config=config,
        run_name=run_name,
        run_dir=run_dir,
        step_callback=store.write_step,
        keep_step_results=False,
    )
    store.write_summary(output.result, output.manager)

    return {
        "run_dir": str(run_dir),
        "steps": len(output.result.steps),
        "submaps": len(output.manager.submaps),
        "global_preview_points": output.result.global_preview_points,
        "final_submap_id": output.result.final_submap_id,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI main。"""

    print(json.dumps(json_safe(run(parse_args(argv))), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

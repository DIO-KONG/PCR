from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.io import read_config
from workflows.pairwise import run_pairwise_task


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a pairwise PCR task: source -> target.")
    parser.add_argument("--task", default="data/tasks/pair_mission1.yaml", help="Path to task YAML/JSON.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to default config YAML/JSON.")
    args = parser.parse_args()

    config = read_config(args.config)
    task = read_config(args.task)
    record = run_pairwise_task(task, config)
    print(f"{record.get('status')}: {record.get('method')} matrix={record.get('matrix_path')}")
    return 0 if record.get("status") in {"success", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

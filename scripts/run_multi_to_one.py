from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.io import read_config
from workflows.multi_to_one import run_multi_to_one_task


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multiple source clouds against one target map.")
    parser.add_argument("--task", default="data/tasks/multi_query_mission1.yaml", help="Path to task YAML/JSON.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to default config YAML/JSON.")
    args = parser.parse_args()

    records = run_multi_to_one_task(read_config(args.task), read_config(args.config))
    for record in records:
        print(f"{record.get('status')}: {record.get('source_id')} -> {record.get('target_id')} free={record.get('is_free')}")
    return 0 if all(record.get("status") in {"success", "skipped"} for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

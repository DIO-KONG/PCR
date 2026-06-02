from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from testbench.standard.run_standard import run_standard
from utils.io import read_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standard PCR benchmark for all registered algorithms.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to default config YAML/JSON.")
    args = parser.parse_args()

    records = run_standard(read_config(args.config))
    for record in records:
        print(f"{record.get('status')}: {record.get('method')} {record.get('matrix_path')}")
    return 0 if all(record.get("status") in {"success", "skipped"} for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

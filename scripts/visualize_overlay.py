from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import io, visualization


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize target and transformed source overlay.")
    parser.add_argument("--source", required=True, help="Source point cloud path.")
    parser.add_argument("--target", required=True, help="Target point cloud path.")
    parser.add_argument("--matrix", required=True, help="4x4 source -> target transform txt.")
    args = parser.parse_args()

    source = io.read_point_cloud(args.source)
    target = io.read_point_cloud(args.target)
    matrix = io.read_matrix(args.matrix)
    visualization.draw_overlay(source, target, matrix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcr.app.sweep_shared_frame_alignment import main


if __name__ == "__main__":
    # 兼容旧命令：
    # .env/bin/python testbench/sweep_shared_frame_alignment.py ...
    #
    # 参数 sweep 的实际实现已经迁移到 `pcr.app.sweep_shared_frame_alignment`，
    # 并复用正式 shared-frame correspondence/RANSAC/scoring 组件。
    main()

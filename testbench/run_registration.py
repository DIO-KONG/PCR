#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcr.app.run_registration import main


if __name__ == "__main__":
    # 兼容旧命令：
    # .env/bin/python testbench/run_registration.py ...
    #
    # Pairwise registration 的实际实现已经迁移到 `pcr.app.run_registration`
    # 和 `pcr.algorithms.shared_frame`，这里不再承载算法或 artifact 写入细节。
    main()

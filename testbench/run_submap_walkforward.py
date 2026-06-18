#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcr.app.run_submap_walkforward import main


if __name__ == "__main__":
    # 兼容 testbench 下的实验入口；真正实现位于 pcr.app / pcr.pipeline。
    main()

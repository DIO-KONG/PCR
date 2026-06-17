#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcr.app.run_walkforward import main


if __name__ == "__main__":
    # 兼容旧命令：
    # .env/bin/python testbench/run_shared_frame_walkforward.py ...
    #
    # 真正的配置解析、pipeline 编排、世界状态更新和 artifact 写入已经迁移到
    # `pcr.app.run_walkforward` 以及 `pcr.pipeline/*`。这里刻意保持薄入口，
    # 避免 testbench 再次长成 God Function。
    main()

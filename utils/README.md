# utils

本目录放置可导入的通用点云工具。

- `preprocess.py`：点云预处理、地板检测、地板对齐、去地板。
- `visualize.py`：Open3D 点云可视化 CLI。
- `pointcloud.py`：点云读写、变换、overlay。
- `metrics.py`：严格配准指标和 gate。

工具模块应保持算法无关，由 `algorithm/` 和 `testbench/` 通过 import 调用。

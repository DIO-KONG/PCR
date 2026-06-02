# specified/

该目录用于放置算法或 pipeline 的针对性实验实现。

建议约定：
- 每个实验保持独立入口。
- 不修改 `data/raw/`。
- 输出到 `results/specified/<algorithm>/<experiment>/<run_id>/`。
- 可复用 `workflows/` 和 `utils/`，不把实验逻辑写入 `algorithms/`。

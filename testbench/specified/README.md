# specified/

该目录用于放置算法或 pipeline 的针对性实验实现。

建议约定：
- 每个实验保持独立入口。
- 不修改 `data/raw/`。
- 输出到 `results/specified/<algorithm>/<experiment>/<run_id>/`。
- 可复用 `workflows/` 和 `utils/`，不把实验逻辑写入 `algorithms/`。

## gicp_refine

当前包含：
- `gicp_refine/random_stability.py`
- `gicp_refine/parameter_robustness.py`
- `gicp_refine/ablation.py`

这些实验只比较 `gicp_refine` 方案和必要 baseline，不新增算法，不改变 algorithms 统一接口。

## multi_comb

当前包含：
- `multi_comb/pair_batch_parameter_test.py`

该实验读取 `data/tasks/pair_batch_incremental_world.yaml`，对每个 pair 测试 rigid、constrained affine、unconstrained affine 三种配置，并生成人工可视化命令列表。

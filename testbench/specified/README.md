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
- `multi_comb/world16_to_world14_parameter_sweep.py`

该实验读取 `data/tasks/pair_batch_incremental_world.yaml`，对每个 pair 测试 rigid、constrained affine、unconstrained affine 三种配置，并生成人工可视化命令列表。
`world16_to_world14_parameter_sweep.py` 只跑 `data/raw/16/world.ply -> data/raw/14/world.ply`，用于覆盖式参数扫描。

当前 `multi_comb` 只做融合前 pairwise 诊断，不更新地图，不做 pose graph。

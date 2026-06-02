# 当前任务

从空目录开始建立 PCR 点云配准项目架构，包含便携环境约束、算法注册、RANSAC baseline、workflow、testbench、utils、CLI、任务配置、结果目录和文档。

## 当前实验计划
- 使用 `ransac_only` 建立第一轮 baseline。
- 默认任务为 `data/tasks/pair_mission1.yaml`。
- source: `data/raw/16_incremental.ply`
- target: `data/raw/14_world.ply`
- 矩阵方向：`P_target ≈ T @ P_source`。
- standard benchmark 每次运行聚合到单一 `results/standard/<run_id>/`。
- pairwise workflow 使用预处理缓存入口。

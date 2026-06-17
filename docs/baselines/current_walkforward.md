# Current Walk-Forward Baseline

本文档记录架构重构后的 walk-forward 功能基线，用于后续继续迁移 pairwise、sweep 或 fusion 策略时做回归对照。

## 命令

```bash
.env/bin/python testbench/run_shared_frame_walkforward.py \
  --config testbench/configs/experiments/shared_frame_walkforward.yaml \
  --run-name arch_refactor_smoke \
  --overwrite
```

旧入口 `testbench/run_shared_frame_walkforward.py` 现在只转调 `pcr.app.run_walkforward`，因此该命令同时验证新架构和旧命令兼容性。

## 输入

- DA3 batch NPZ：`da3/data/raw/pointcloud/batch_*.npz`
- DA3 raw PLY：`da3/data/raw/pointcloud/batch_*.ply`
- 预处理输出：`result/preprocess/<batch>/floor_removed.ply`
- 配置：`testbench/configs/experiments/shared_frame_walkforward.yaml`

如果 `floor_removed.ply` 缺失，`pcr.preprocessing.PreprocessService` 会调用 `utils.preprocess` 自动补齐。

## 关键输出

输出目录：

```text
result/walkforward_shared_frame/arch_refactor_smoke/
```

关键 artifact：

- `final_world.ply`
- `summary.json`
- `summary.md`
- `transforms/*_to_global.txt`
- 每步 `coarse_overlay.ply`
- 每步 `bounded_icp_overlay.ply`
- 每步 `final_matrix.txt`
- 每步 `metrics.json`

## 本次验证结果

```text
final_world_points: 90608
step 01: frames=9,11,12   icp=accepted             dt=0.1445m drot=4.643deg
step 02: frames=10,11,12  icp=accepted             dt=0.0970m drot=2.316deg
step 03: frames=12,13,14  icp=rejected_by_boundary dt=0.2939m drot=2.567deg
step 04: frames=13,14,15  icp=accepted             dt=0.1258m drot=2.745deg
step 05: frames=13,14,15  icp=accepted             dt=0.0306m drot=0.835deg
step 06: frames=15,16,17  icp=accepted             dt=0.0900m drot=2.610deg
```

第 3 步 ICP 被边界拒绝，符合当前 bounded ICP 策略：超过 `0.15m / 5deg` 中的平移边界时回退 coarse。

## 验收标准

后续重构不要求 PLY 字节完全一致，但应满足：

- 旧 walk-forward 命令仍可执行。
- 6 个 step 都能生成 `metrics.json`、`final_matrix.txt`、overlay 和 `fused_world.ply`。
- `transforms/` 中包含 baseline 和 6 个 window 的 `_to_global.txt`。
- `final_world.ply` 非空。
- `summary.json` 能读出每步 selected frames、ICP 状态、ICP delta 和最终点数。


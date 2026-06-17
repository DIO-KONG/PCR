# testbench

本目录放置实验配置和实验入口。

当前实验默认运行 `shared_frame_alignment`。它使用 DA3 共享帧估计第一组 window 到 baseline 的刚体变换，可通过实验入口运行：

```bash
.env/bin/python testbench/run_registration.py \
  --scheme shared_frame_alignment \
  --run-name shared_frame_step01_refined \
  --overwrite
```

配置结构：

- `configs/datasets/`：数据集 source/target 配置。
- `configs/algorithms/`：单个算法参数。
- `configs/experiments/`：组合多个算法方案的实验配置。

实验输出默认写入 `result/registration_schemes/`，该目录由 `.gitignore` 忽略。

常用检查文件：

- `matrix.txt`：最终选中的 4x4 刚体矩阵。
- `metrics.json`：完整指标和候选信息。
- `top_candidates.md`：多个候选的简表。
- `overlay.ply`：target/source 配准效果叠加可视化。

## 共享帧粗配准参数 sweep

`sweep_shared_frame_alignment.py` 用于快速筛选共享帧粗配准参数。它只读取 DA3 NPZ，不依赖 `result/preprocess` 中的去地板点云；默认关闭 ICP，不生成 overlay。

```bash
.env/bin/python testbench/sweep_shared_frame_alignment.py \
  --run-name conf_stride_frame3_threshold_sweep
```

默认会测试：

- `sampling.conf_percentile = 0, 5, 10, 20`
- `sampling.stride = 6, 4, 3, 2`
- 自动共享帧的 4选3 组合，以及全共享帧组合
- `ransac.thresholds = 0.18, 0.14, 0.10, 0.06`
- 每组默认 `300` 次 RANSAC，用于快速筛选趋势；候选收敛后再提高迭代复跑。

结果写入 `result/shared_frame_sweep/<run_name>/`。

## Walk-Forward 建图

`run_shared_frame_walkforward.py` 用于运行完整的相邻 batch walk-forward 建图实验。

```bash
.env/bin/python testbench/run_shared_frame_walkforward.py \
  --config testbench/configs/experiments/shared_frame_walkforward.yaml \
  --run-name dynamic_top3_bounded_icp_walkforward \
  --overwrite
```

该入口会自动补齐缺失的 `result/preprocess/<batch>/floor_removed.ply`，然后按配置中的显式 step 链执行：

```text
batch_02 -> batch_01
batch_03 -> batch_02
batch_04 -> batch_03
batch_05 -> batch_04
batch_06 -> batch_05
batch_07 -> batch_06
```

每步输出 frame 组合排序、coarse overlay、bounded ICP overlay、最终矩阵和融合后的 world。总结果位于 `result/walkforward_shared_frame/<run_name>/`。

# PCR

本项目在 WSL Ubuntu 22.04 中开发，所有 Python 环境和依赖都位于 `.env/`。

## 可视化点云

```bash
.env/bin/python utils/visualize.py da3/data/raw/pointcloud/batch_01_baseline_001-012.ply
```

两个点云对比可视化：

```bash
.env/bin/python utils/visualize.py target.ply source.ply
```

## 点云预处理

`utils/preprocess.py` 同时提供 Python API 和 CLI。模块只做点云预处理，不修改 `da3/` 原始数据。

基础统计：

```bash
.env/bin/python utils/preprocess.py da3/data/raw/pointcloud/batch_01_baseline_001-012.ply --print-stats
```

地板法线对齐到 `+Y`，并保存调试产物：

```bash
.env/bin/python utils/preprocess.py da3/data/raw/pointcloud/batch_01_baseline_001-012.ply \
  --output result/preprocess/batch_01/aligned.ply \
  --align-floor \
  --write-debug result/preprocess/batch_01/debug
```

地板对齐后去地板：

```bash
.env/bin/python utils/preprocess.py da3/data/raw/pointcloud/batch_01_baseline_001-012.ply \
  --output result/preprocess/batch_01/floor_removed.ply \
  --align-floor \
  --remove-floor \
  --write-debug result/preprocess/batch_01_removed/debug
```

如果不显式指定 `--output`，只要执行了会产生新点云的处理步骤，CLI 会默认写入项目内基准目录：

```text
result/preprocess/<输入文件stem>/
```

例如：

```bash
.env/bin/python utils/preprocess.py da3/data/raw/pointcloud/batch_01_baseline_001-012.ply \
  --align-floor \
  --write-debug
```

会生成：

```text
result/preprocess/batch_01_baseline_001-012/aligned.ply
result/preprocess/batch_01_baseline_001-012/debug/
```

地板检测不使用固定 `y_min/y_max` 高度阈值。当前策略是先迭代提取多个 RANSAC 平面，再根据法线是否接近 `Y` 轴、点数支持和面积估计选择最可信的地板候选。如果没有可信地板，CLI 会输出 `floor_not_found`，不会静默误删墙面或隔板。

## 点云配准框架

当前框架把通用工具放在 `utils/`，配准算法放在 `algorithm/`，实验入口和配置放在 `testbench/`。

当前已实现第一个可控算法 `shared_frame_alignment`。它不是传统点云特征配准，而是利用 DA3 batch 之间共享 RGB-D 帧的同像素 3D 对应关系，估计 window batch 到 baseline batch 的刚体变换。当前实现保持刚体配准，不做 Sim(3) 尺度微调。

这个算法适合当前第一步场景：`batch_02_window_009-013` 与 `batch_01_baseline_001-012` 共享 `9-rgb.png` 到 `12-rgb.png` 四帧。算法先把共享帧中同一像素通过各自 batch 的深度、内参、外参反投影到两个 batch 的局部 world，再用 RANSAC + Kabsch/SVD 估计 `window -> baseline` 的刚体矩阵。为了降低 DA3 置信像素差异、局部深度噪声和少量错误对应点的影响，当前版本会从 `0.12m`、`0.10m`、`0.08m`、`0.06m` 多个粗阈值生成候选，然后用逐步收紧的刚体 refit 精修，最终统一按 `0.06m` 统计指标。候选选定后还会执行一次有边界的 point-to-plane ICP，只允许小幅刚体微调；如果相对初值移动超过 `0.15m`、旋转超过 `5deg`，或明显破坏共享帧一致性，则自动回退。

运行第一步：

```bash
.env/bin/python testbench/run_registration.py \
  --scheme shared_frame_alignment \
  --run-name shared_frame_step01_refined \
  --overwrite
```

后续新增算法时需要：

1. 在 `algorithm/` 下新增算法实现。
2. 在 `algorithm/registry.py` 中注册算法名。
3. 在 `testbench/configs/algorithms/` 下新增对应 YAML。
4. 在 `testbench/configs/experiments/registration_schemes.yaml` 的 `schemes` 中加入该 YAML 路径。

默认输入为：

```text
source: da3/data/raw/pointcloud/batch_02_window_009-013.ply
target: da3/data/raw/pointcloud/batch_01_baseline_001-012.ply
```

输出位于：

```text
result/registration_schemes/<run_name>/<scheme_name>/
```

每个方案会保存：

```text
matrix.txt
metrics.json
top_candidates.md
registered_source.ply
overlay.ply
```

重点查看：

- `overlay.ply`：灰色为 target，黄色为变换后的 source。
- `matrix.txt`：最终选中的刚体矩阵。
- `metrics.json`：包含整体误差、每个共享帧的误差、RANSAC/refit 历史。
- `top_candidates.md`：多个粗阈值候选的简表；walk-forward 或人工检查时只取排序第一的候选。

`shared_frame_alignment` 默认使用：

```text
source_npz: da3/data/raw/pointcloud/batch_02_window_009-013.npz
target_npz: da3/data/raw/pointcloud/batch_01_baseline_001-012.npz
shared frames: 9-rgb.png, 10-rgb.png, 11-rgb.png, 12-rgb.png
overlay clouds: result/preprocess/.../floor_removed.ply
```

算法详细说明见 `algorithm/shared_frame_alignment.md`。

## 共享帧粗配准参数实验

如果只想评估共享帧粗配准参数，不生成 overlay，也不跑 ICP，可使用 sweep 入口：

```bash
.env/bin/python testbench/sweep_shared_frame_alignment.py \
  --run-name conf_stride_frame3_threshold_sweep
```

该入口会自动从两个 DA3 NPZ 中发现共享帧，不硬编码 `9/10/11/12`。默认实验组合为：

```text
sampling.conf_percentile: 0, 5, 10, 20
sampling.stride:          6, 4, 3, 2
shared frame sets:        自动共享帧 4选3 + 全共享帧
ransac.thresholds:        0.18, 0.14, 0.10, 0.06
ransac.iterations:        300
```

输出位于：

```text
result/shared_frame_sweep/<run_name>/
```

重点查看：

```text
summary.md
summary.csv
summary.json
```

## Walk-Forward 建图实验

当前推荐的完整链路是 `dynamic_top3_shared_frame_bounded_icp`：

```bash
.env/bin/python testbench/run_shared_frame_walkforward.py \
  --config testbench/configs/experiments/shared_frame_walkforward.yaml \
  --run-name dynamic_top3_bounded_icp_walkforward \
  --overwrite
```

该入口会：

- 自动补齐 `result/preprocess/<batch>/floor_removed.ply`。
- 对每个相邻 window 自动发现共享帧并做 4选3。
- 使用最佳 3 帧组合做共享帧刚体粗配准。
- 将 source 投到累计 world 后，对累计 world 做 bounded point-to-plane ICP。
- ICP 在 `0.15m / 5deg` 内即采用，否则回退 coarse。
- 每步融合进累计 world，并保存最终 `final_world.ply`。

输出位于：

```text
result/walkforward_shared_frame/<run_name>/
```

重点查看：

```text
summary.md
summary.json
final_world.ply
step_*/frame_combo_ranking.md
step_*/coarse_overlay.ply
step_*/bounded_icp_overlay.ply
step_*/metrics.json
```

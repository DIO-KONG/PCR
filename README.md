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

## 多方案点云配准

当前框架把通用工具放在 `utils/`，配准算法放在 `algorithm/`，实验入口和配置放在 `testbench/`。

运行所有方案：

```bash
.env/bin/python testbench/run_registration.py \
  --run-name current_da3_pairwise \
  --overwrite
```

只运行一个方案：

```bash
.env/bin/python testbench/run_registration.py \
  --scheme fpfh_ransac_icp \
  --run-name current_da3_icp \
  --overwrite
```

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
registered_source_preprocessed.ply
overlay_preprocessed.ply
```

当前方案包括：

- `fpfh_ransac`
- `fpfh_ransac_icp`
- `fpfh_colored_icp`
- `teaser_icp`

`teaser_icp` 需要额外安装 `teaserpp_python`。如果当前 `.env` 中没有该依赖，实验会把该方案标记为 `skipped_missing_teaser_dependency`。

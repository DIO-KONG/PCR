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

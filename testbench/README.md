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

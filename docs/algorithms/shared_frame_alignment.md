# shared_frame_alignment

`shared_frame_alignment` 是当前第一阶段的 DA3 batch 对齐算法。它不使用传统点云全局特征，也不做尺度优化，而是利用两个 batch 中真实共享的 RGB-D 帧建立同像素 3D 对应关系，再估计 source batch 到 target batch 的刚体变换。

## 适用场景

当前第一步配准为：

```text
source: batch_002_window_009-013
target: batch_001_baseline_001-012
shared frames: 9.png, 10.png, 11.png, 12.png
```

这两个 batch 对同一批图像帧分别预测了深度、内参、外参。虽然两个 batch 的局部 world 坐标系不同，但同一图像、同一像素理论上对应同一个真实空间点。因此可以把共享像素分别反投影到 source world 和 target world，形成 `source_point -> target_point` 的 3D 对应。

长序列数据中图像命名为 `9.png`、`10.png` 等；实现按文件名前缀数字排序，并兼容历史 `9-rgb.png` 命名。

## 算法步骤

1. 读取 `source_npz` 和 `target_npz`。
2. 找到配置中的共享帧。
3. 对每个共享帧，保留两个 batch 中深度有效、置信度足够高的同像素点。
4. 使用各自的深度、内参、外参将像素反投影到两个 batch 的局部 world。
5. 使用多个 RANSAC 距离阈值生成多个刚体候选。
6. 每个候选通过 Kabsch/SVD 在内点上做刚体 refit。
7. 使用逐步收紧的阈值再次 refit，只估计旋转和平移，不估计尺度。
8. 所有候选统一按严格 `evaluation_threshold` 重新计算误差并排序。
9. 对排序第一的候选执行有边界 refine。默认是 point-to-plane ICP；调试时也可切换为 Sim(3) point-to-point refine，用于验证 DA3 batch 间轻微尺度差异的影响。
10. 输出 top K 候选，实验流程默认选排序第一的候选。

## 尺度调试

DA3 不同 batch 之间可能存在轻微尺度差异，例如 5% 左右。但当前阶段先保持刚体配准，有两个原因：

- 便于判断误差主要来自姿态、深度噪声、像素覆盖差异，还是确实需要 Sim(3)。
- 避免尺度自由度在局部共享帧上吸收错误对应点，导致视觉上更难解释的变形。

当前已提供一个有边界 Sim(3) 对照 refine：

```yaml
algorithm:
  icp:
    method: sim3_point_to_point
    max_scale_delta: 0.08
```

粗配准仍然只估计 SE(3)，Sim(3) 只在 refine 阶段生效。若 refine 相对 coarse 的平移、旋转或尺度变化超过边界，会回退到 coarse。`metrics.json` 中会记录 `delta_scale` 和 `scale_delta`。

## 关键参数

`sampling.conf_percentile` 表示丢弃低置信度像素的百分位。当前 walk-forward 配置为 `20.0`，即保留较高置信度的 80% 像素点。参数 sweep 中仍会测试 `0/5/10/20`，用于观察采样密度和噪声之间的权衡。

`ransac.thresholds` 是粗候选生成阈值。当前为：

```text
0.18m, 0.14m, 0.10m, 0.06m
```

较宽阈值用于在 DA3 深度和置信像素不完全一致时找到稳定初值；最终仍用严格阈值评估。

`ransac.evaluation_threshold` 是统一评价阈值，当前为 `0.06m`。这表示误差小于 6cm 的共享像素点计为 inlier。

`refinement.thresholds` 是逐步收紧 refit 的阈值。每次 refit 都是刚体 Kabsch/SVD，不允许尺度变化。

`icp_refinement` 是可选局部刚体精修。当前默认启用，主要参数为：

```text
voxel_size: 0.06m
max_correspondence_distance: 0.08m
max_translation_delta: 0.15m
max_rotation_delta_deg: 5deg
```

如果 ICP 结果相对共享帧初值移动超过边界，算法会拒绝 ICP 结果并回退到 ICP 前的矩阵。即使运动边界通过，算法还会检查共享帧误差是否明显变差；如果 6cm inlier ratio 大幅下降，或 median error 明显升高，也会拒绝 ICP。这个守门是为了避免办公室重复隔板结构把 ICP 拉到视觉上相似但共享帧不一致的位置。

## 输出指标

`metrics.json` 中最重要的字段：

- `inlier_ratio`：严格阈值下，误差小于 `evaluation_threshold` 的对应点占比。
- `median_error`：共享像素 3D 对应误差的中位数，单位为米。
- `p90_error`：90 分位误差，单位为米。
- `per_frame_metrics`：按共享帧拆开的误差统计，用于发现某一帧明显拖累整体估计。
- `refinement_history`：逐阈值 refit 的过程记录。
- `icp_refinement`：局部 ICP 的 fitness、RMSE、相对初值位移和旋转，以及是否被边界接受。
- `pre_icp_shared_metrics`：进入 ICP 前的共享帧对应点指标，便于对比 ICP 是否改善视觉点云但损伤共享帧误差。

每个 step 还会输出 `shared_frame_overlay.ply`：

- target/baseline 共享帧点：灰色。
- source/window 共享帧点：先用 coarse source->target 变换投到 target 坐标，再染成黄色。

这个文件用于调试粗配准本身，不受 global world、ICP target 或融合策略影响。

`top_candidates.md` 中的候选来自不同粗阈值。候选排序分数只用于实验选择，不代表真实建图质量；最终仍应结合 `overlay.ply` 可视化检查。

## 参数 sweep

粗配准参数筛选入口位于：

```text
testbench/sweep_shared_frame_alignment.py
```

该脚本会自动从两个 DA3 NPZ 的 `image_names` 中发现共享帧，并生成共享帧组合，不硬编码某个 window 中具体哪几帧表现好。默认用于测试：

- `sampling.conf_percentile = 0, 5, 10, 20`
- `sampling.stride = 6, 4, 3, 2`
- 自动共享帧的 4选3 组合和全共享帧组合
- `ransac.thresholds = 0.18, 0.14, 0.10, 0.06`

这个 sweep 默认关闭 ICP，只评估共享帧粗配准本身。原因是 ICP 会引入点云局部表面几何因素，容易掩盖采样和共享帧组合对粗配准的影响。

## dynamic_top3_shared_frame_bounded_icp

`dynamic_top3_shared_frame_bounded_icp` 是当前用于 walk-forward 建图的版本。它把前面单步实验中确认较稳定的策略固定下来：

- 自动发现相邻 DA3 batch 的共享帧。
- 枚举共享帧 4选3 组合，不硬编码某个 window 中哪几帧好。
- 对每个 3 帧组合执行共享像素 3D 对应点粗配准。
- 按 `score = inlier_ratio - median_error - 0.25 * p90_error` 选择最佳组合。
- 将相邻 batch 刚体变换组合到 global 坐标系。
- 以累计 world 为 target 做有边界 point-to-plane ICP。
- ICP 在 `0.15m / 5deg` 边界内即采用；共享帧一致性只记录为诊断，不作为拒绝条件。

当前实现已经拆到新的 `pcr/` 架构中：

- `pcr.algorithms.shared_frame.frame_selection`：动态共享帧发现、4选3、候选排序。
- `pcr.algorithms.refinement.icp`：有边界 point-to-plane ICP。
- `pcr.pipeline.registration_step`：单步 coarse + ICP 编排。
- `pcr.state.world`：累计 world 和 batch 到 global 的 transform graph。
- `pcr.artifacts.store`：矩阵、overlay、metrics、summary 写入。

旧入口 `testbench/run_shared_frame_walkforward.py` 只保留命令兼容，不再承载算法细节。

固定参数为：

```text
sampling.conf_percentile: 20.0
sampling.stride: 6
sampling.max_points_per_frame: 8000
ransac.thresholds: 0.18, 0.14, 0.10, 0.06
ransac.iterations: 3000
ransac.evaluation_threshold: 0.06
icp.voxel_size: 0.06
icp.max_correspondence_distance: 0.08
icp.max_translation_delta: 0.15
icp.max_rotation_delta_deg: 5.0
fusion.voxel_size: 0.06
```

完整 walk-forward 入口为：

```bash
.env/bin/python testbench/run_shared_frame_walkforward.py \
  --config testbench/configs/experiments/shared_frame_walkforward.yaml \
  --run-name dynamic_top3_bounded_icp_walkforward \
  --overwrite
```

输出位于：

```text
result/walkforward_shared_frame/<run_name>/
```

## online_submap_mapping 中的角色

300 步级别建图时，`shared_frame_alignment` 仍只负责局部刚体位姿：

- source window 到 target batch 的 shared-frame coarse。
- source window 到 active submap 的 bounded ICP 初值和微调。

它不负责融合。融合由 `ConservativeVoxelHashFusion` 处理，并且只融合当前
window 的非共享新增帧，避免共享帧在多个 batch 中被重复写入地图。完整流程见：

```text
docs/algorithms/online_submap_mapping.md
```

## 已知局限

- 同一像素在不同 batch 中的置信度和深度可能不同，导致对应点含有系统噪声。
- 如果 DA3 两个 batch 的外参存在尺度差异，刚体算法无法完全消除误差。
- 如果共享帧中可见结构大面积重复，RANSAC 仍可能保留局部错误内点。
- 当前算法只解决 batch 坐标系对齐，不直接处理后续融合中的重影、密度不一致和地图更新策略。

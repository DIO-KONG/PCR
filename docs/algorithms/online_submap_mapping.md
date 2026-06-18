# online_submap_mapping

`online_submap_mapping` 面向“机器人走一步等一步”的主动寻路场景。它不是离线一次性重建，而是每处理一个 5 图 window 就更新当前局部 submap，并维护 submap 之间的位姿链。

## 数据生成

输入图像为：

```text
da3/data/raw/image/1.png ... da3/data/raw/image/327.png
```

DA3 batch 规则：

- `batch_001_baseline_001-012` 使用 `1..12` 初始化地图。
- 从第 13 帧开始，每个 window 使用当前帧和前 4 帧。
- 最后一个 batch 是 `batch_316_window_323-327`。

先批量生成 DA3 结果：

```bash
.env/bin/python da3/run_da3_batches.py \
  --image-dir da3/data/raw/image \
  --output-dir da3/data/raw/pointcloud \
  --frame-start 1 \
  --frame-end 327 \
  --baseline-size 12 \
  --window-size 5 \
  --conf-percentile 20 \
  --skip-existing
```

`.npz` 是后续 frame-level 融合的主数据源，batch `.ply` 用于预处理和人工检查。

## 在线流程

每一步执行：

1. 自动发现 source window 与 target batch 的共享帧。
2. 枚举共享帧 `4选3`，用共享像素 3D 对应点做刚体粗配准。
3. 将 `T_window_to_reference` 组合成 `T_window_to_active_submap`。
4. 以 active submap 的 local cloud 为 target 执行 bounded point-to-plane ICP。
5. 运行质量门控，检查 shared-frame 误差、ICP RMSE、ICP fitness 和 ICP 位移/旋转边界。
6. 从 source `.npz` 中只生成非共享新增帧点云，例如 `9..13` 只融合 `13.png`。
7. 将新增帧点云变换到 active submap，并用 Voxel Hash 做保守融合。
8. active submap 满 10 个 window 后，使用最近 3 个 batch 作为 overlap seed 创建下一个 submap。

## Voxel Hash 融合

融合目标是保守建图，而不是把所有点平均进去：

- 新区域点写入主 voxel map。
- 与已有地图距离和法线一致的点只更新体素，不增加密度。
- 距离接近但法线或位置冲突的点写入 conflict debug，不污染主地图。

默认参数：

```text
voxel_size: 0.06
duplicate_distance: 0.04
conflict_distance: 0.15
normal_angle_deg: 25
```

每步保存：

```text
accepted_points.ply
duplicate_points.ply
conflict_points.ply
fusion_report.json
quality_report.json
```

## Submap 位姿链

`submap_000` 以 baseline 为原点：

```text
T_submap_000_to_global = I
```

后续每个 submap 记录：

```text
T_submap_i_to_submap_i-1
T_submap_i_to_global = T_submap_i_to_submap_i-1 @ T_submap_i-1_to_global
```

第一版只维护链式位姿，不做回环和 pose graph 全局优化。显示全局预览时，将每个 submap 的 `local_world.ply` 按链式位姿临时组合成 `global_preview.ply`。

## 运行

生成显式 327 帧配置：

```bash
.env/bin/python -m pcr.app.generate_submap_config \
  --output testbench/configs/experiments/submap_walkforward_327.yaml
```

运行 online submap：

```bash
.env/bin/python testbench/run_submap_walkforward.py \
  --config testbench/configs/experiments/submap_walkforward_327.yaml \
  --run-name submap_327 \
  --overwrite
```

输出位于：

```text
result/submap_walkforward/<run_name>/
```

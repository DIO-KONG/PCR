# PCR: DA3 点云生成与配准建图实验框架

这个项目用于从 Depth Anything 3 生成局部点云，并在不依赖 DA3 外参初值的前提下，做点云配准、walk-forward 融合建图和参数实验。

当前工作流分成两段：

1. 使用 DA3 从 `da3/data/raw/image/` 中的 RGB 图像生成每个 batch 的深度、内参、外参和融合点云。
2. 使用 `utils/`、`algorithm/`、`testbench/` 组成的实验框架，对 batch 点云做基础几何配准与 walk-forward 融合，输出到 `result/`。

## 项目结构

```text
.
├── CODEX.md
├── README.md
├── algorithm/
│   ├── fpfh_ransac_icp.py
│   ├── registry.py
│   └── topview_vote.py
├── da3/
│   ├── run_da3_batches.py
│   └── data/
│       └── raw/
│           ├── image/
│           └── pointcloud/
├── testbench/
│   ├── run_experiment.py
│   └── configs/
│       ├── algorithms/
│       ├── datasets/
│       └── experiments/
├── utils/
│   ├── config.py
│   ├── data.py
│   ├── fusion.py
│   ├── io.py
│   ├── metrics.py
│   ├── pointcloud.py
│   ├── preprocess.py
│   └── types.py
└── result/
```

目录职责：

- `da3/`：DA3 batch 推理和原始数据区。
- `da3/data/raw/image/`：输入 RGB 图像，文件名形如 `1-rgb.png`。
- `da3/data/raw/pointcloud/`：DA3 生成的 `.npz` 和 `.ply`，作为配准实验的只读输入。
- `utils/`：通用工具，负责配置读取、点云 I/O、预处理、指标、融合和结果保存。
- `algorithm/`：配准算法代码。
- `testbench/`：实验配置和运行入口。
- `result/`：实验输出目录，是派生产物，不纳入版本控制。

## 环境约束

所有 Python 环境和依赖必须安装到 `.env/` 中，不要安装到系统 Python 或 user site-packages。

当前项目使用的核心依赖包括：

- Python 3.10+
- Depth Anything 3
- PyTorch
- Open3D
- NumPy
- PyYAML

如果 `.env/` 已存在，可直接使用：

```bash
.env/bin/python --version
```

如果需要重建环境，使用 Ubuntu/Python 工具创建虚拟环境，并将依赖安装到 `.env/`。DA3 与模型权重下载可能需要网络。国内网络环境建议使用 Hugging Face 镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 .env/bin/python ...
```

## DA3 点云生成

当前 DA3 脚本是：

```text
da3/run_da3_batches.py
```

它会读取：

```text
da3/data/raw/image/{1..18}-rgb.png
```

并生成固定 7 个 batch：

| batch | frames | output stem |
|---|---|---|
| baseline | `1..12` | `batch_01_baseline_001-012` |
| window | `9..13` | `batch_02_window_009-013` |
| window | `10..14` | `batch_03_window_010-014` |
| window | `11..15` | `batch_04_window_011-015` |
| window | `12..16` | `batch_05_window_012-016` |
| window | `13..17` | `batch_06_window_013-017` |
| window | `14..18` | `batch_07_window_014-018` |

每个 batch 会保存：

- `.npz`：包含 `image_names`、`depth`、`conf`、`intrinsics`、`extrinsics`、`processed_images`。
- `.ply`：由该 batch 深度图融合出的 Open3D 点云。

运行 dry-run 检查图像和 batch：

```bash
.env/bin/python da3/run_da3_batches.py --dry-run
```

运行完整 DA3 推理：

```bash
env PYTHONUNBUFFERED=1 \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DISABLE_XET=1 \
  .env/bin/python da3/run_da3_batches.py
```

默认 DA3 模型：

```text
depth-anything/DA3NESTED-GIANT-LARGE-1.1
```

注意：该模型要求 CUDA。若 CUDA 不可用，脚本会直接失败，而不是静默切到 CPU。

## 配准建图实验框架

实验框架不使用 DA3 外参作为配准初值。当前 v1 只基于 `.ply` 点云几何做基础配准。

### 代码注释要求

所有项目代码都需要有详细中文注释。注释重点不是逐行翻译语法，而是解释：

- 该模块在建图流程中的职责。
- 几何假设，例如矩阵方向、坐标轴、尺度含义。
- 配准、融合、指标计算中的关键阈值为什么存在。
- 非显然逻辑，例如 top-view 候选生成、translation voting、重影过滤。
- 未来维护者调参时应该注意的风险。

### 融合方案路线

当前仅实现“重影检测 / 冲突过滤”，后续可以按效果逐步升级：

1. Voxel / Hash Map 融合
   - 每个 voxel 维护一个稳定代表点、颜色、观测次数和权重。
   - 新点进入已有 voxel 时做加权更新，而不是无限追加点。
   - 目标是保持密度不随观测次数膨胀。
2. 重影检测 / 冲突过滤
   - 当前已实现。
   - 配准后，将 source 点变换到当前 world 坐标系。
   - 如果 source 点到地图最近邻距离低于 `duplicate_distance`，认为地图中已经存在该表面，丢弃以保持密度稳定。
   - 如果距离介于 `duplicate_distance` 和 `conflict_distance` 之间，认为它靠近已有表面但未对齐，作为潜在重影丢弃。
   - 如果距离高于 `conflict_distance`，认为它可能是新增区域，允许单独加入地图。
   - 类似地，`9..13` 的点云经过配准得到变换后，可以检测其新增部分是否已存在于地图中；若新增区域与地图距离高于阈值，则按该变换单独加入对应新区域。
3. Surfel 融合
   - 如果“重影检测 / 冲突过滤”效果不理想，再考虑。
   - 每个 surfel 维护 position、normal、color、radius、confidence、timestamp。
   - 适合更细粒度地处理重复观测、法线冲突和局部表面更新。
4. Pose Graph / Submap，而不是立即硬融合
   - 如果地图变大或长序列累计漂移明显，应优先保留 submap 和 transform。
   - 先做子图级约束和全局优化，再统一导出地图。
   - 这样可以避免某一步错误融合污染后续所有 world。

### 当前数据集

数据集配置：

```text
testbench/configs/datasets/current_da3.yaml
```

当前 walk-forward 链：

```text
batch_02_window_009-013 -> batch_01_baseline_001-012 -> world_013
batch_03_window_010-014 -> world_013 -> world_014
batch_04_window_011-015 -> world_014 -> world_015
batch_05_window_012-016 -> world_015 -> world_016
batch_06_window_013-017 -> world_016 -> world_017
batch_07_window_014-018 -> world_017 -> world_018
```

`da3/data/raw/pointcloud/` 只作为输入，不写入任何配准实验产物。

### 算法

当前有两个算法：

#### `topview_vote`

配置：

```text
testbench/configs/algorithms/topview_vote.yaml
```

这是当前主算法。它只使用点云几何：

1. 对 source 和 target 做 SOR、体素降采样、法线估计。
2. 将点云投影到 XZ 平面，忽略高度 Y。
3. 枚举 yaw 和水平尺度，生成 top-view occupancy 相关性候选。
4. 对候选按旋转聚类，选出需要精修的候选。
5. 对候选执行水平/垂直尺度搜索和 translation voting。
6. 用 nearest-neighbor 指标评估候选并排名。
7. 保存 top candidates、矩阵、metrics 和 overlay。

输出矩阵方向固定为：

```text
P_target ~= T @ P_source
```

注意：`topview_vote` 允许矩阵包含尺度，不是严格 SE(3)。它是几何校正候选，不应直接当作机器人里程计 pose。

#### `fpfh_ransac_icp`

配置：

```text
testbench/configs/algorithms/fpfh_ransac_icp.yaml
```

这是对照算法：

1. Open3D FPFH 特征。
2. RANSAC 粗配准。
3. point-to-plane ICP 精修。

它用于 benchmark 和 sanity check，不是默认融合算法。

## 运行实验

统一入口：

```bash
.env/bin/python testbench/run_experiment.py
```

默认实验配置：

```text
testbench/configs/experiments/basic_walk_forward.yaml
```

### 1. 验证数据集

检查配置中的 PLY 是否存在、Open3D 是否能加载、点数是否非零：

```bash
.env/bin/python testbench/run_experiment.py \
  --mode validate \
  --run-name validate_current \
  --overwrite
```

输出：

```text
result/validate_current/
├── run_config.yaml
└── validation.json
```

### 2. 单步 pairwise 配准

默认会跑数据集里的第一步：

```bash
.env/bin/python testbench/run_experiment.py \
  --mode pairwise \
  --run-name pairwise_topview_smoke \
  --overwrite
```

输出：

```text
result/pairwise_topview_smoke/
├── run_config.yaml
├── summary.json
├── summary.md
└── steps/
    └── world_013/
        ├── matrix.txt
        ├── metrics.json
        ├── overlay_top1.ply
        ├── registered_source.ply
        └── top_candidates.md
```

### 3. 完整 walk-forward 融合

```bash
.env/bin/python testbench/run_experiment.py \
  --mode walk_forward \
  --run-name walk_forward_topview_smoke \
  --overwrite
```

输出：

```text
result/walk_forward_topview_smoke/
├── final_world.ply
├── run_config.yaml
├── summary.json
├── summary.md
└── steps/
    ├── 01_world_013/
    ├── 02_world_014/
    ├── 03_world_015/
    ├── 04_world_016/
    ├── 05_world_017/
    └── 06_world_018/
```

每个 step 目录包含：

- `matrix.txt`：source 到 target/world 的 4x4 变换。
- `metrics.json`：指标、状态、top candidates。
- `fusion_stats.json`：融合阶段的重复点、冲突点、新增点统计。
- `registered_source.ply`：变换后的 source。
- `overlay_top1.ply`：target/world 灰色，source 黄色。
- `top_candidates.md`：候选排名表。
- `fused_world.ply`：walk-forward 模式下的融合地图。
- `fusion_debug/accepted_new_points.ply`：通过冲突过滤后真正加入地图的新点。
- `fusion_debug/rejected_conflict_points.ply`：被判定为潜在重影冲突并丢弃的点。
- `fusion_debug/duplicate_points.ply`：被判定为地图中已有表面的重复点。

## 结果怎么看

优先查看：

1. `summary.md`：快速浏览每步的 fitness、trimmed distance、scale 和 fused 点数。
2. `steps/*/overlay_top1.ply`：人工检查配准是否视觉正确。
3. `steps/*/top_candidates.md`：当 top1 视觉不对时，查看其他候选。
4. `final_world.ply`：查看最终融合地图。

质量状态含义：

- `accepted_by_numeric_gate`：数值门槛未发现明显问题，但不等于视觉正确。
- `needs_review_low_fitness`：重叠比例低，需要检查。
- `needs_review_high_trimmed`：最近邻 trimmed distance 偏大，需要检查。
- `needs_review_scale`：尺度超出配置门槛，需要检查。

当前框架的质量门只做自动标记，不会阻止融合。这样便于实验阶段保留完整结果链。

## 配置说明

实验配置由三层组成：

```text
experiment -> dataset_config + algorithm_config + algorithm_overrides
```

例子：

```yaml
name: basic_walk_forward
dataset_config: testbench/configs/datasets/current_da3.yaml
algorithm_config: testbench/configs/algorithms/topview_vote.yaml
algorithm_overrides: {}
```

如果要改参数但不想复制整份算法配置，可以在 `algorithm_overrides` 中覆盖局部字段：

```yaml
algorithm_overrides:
  registration:
    topview_yaw_step_deg: 2.5
    max_refine_candidates: 40
  fusion:
    fusion_voxel_size: 0.05
```

### 多组融合参数 sweep

当前实验配置提供了 `fusion_sweeps`，用于一次运行多组“重影检测 / 冲突过滤”阈值：

```bash
.env/bin/python testbench/run_experiment.py \
  --mode fusion_sweep \
  --run-name fusion_conflict_sweep \
  --overwrite
```

输出结构：

```text
result/fusion_conflict_sweep/
├── 01_conflict_strict/
├── 02_conflict_balanced/
├── 03_conflict_loose/
├── summary.json
└── summary.md
```

每个子目录都有自己的 `final_world.ply`，用于可视化择优。

## 新增数据集

新增数据时，推荐创建新的 dataset YAML，例如：

```text
testbench/configs/datasets/my_new_sequence.yaml
```

格式：

```yaml
name: my_new_sequence
root: .
initial_world: path/to/baseline_world.ply
steps:
  - name: world_020
    source: path/to/incremental_020.ply
    output_world: world_020
  - name: world_022
    source: path/to/incremental_022.ply
    output_world: world_022
```

原则：

- 用显式 step 列表描述链路。
- 不从文件名隐式推断顺序。
- raw 数据保持只读。
- 输出总是写到 `result/`。

## 新增算法

新增算法时：

1. 在 `algorithm/` 下新增文件，例如 `algorithm/my_method.py`。
2. 实现统一接口：

```python
def register(source_path, target_path, config) -> RegistrationResult:
    ...
```

3. 在 `algorithm/registry.py` 中注册：

```python
from algorithm.my_method import register as my_method

ALGORITHMS = {
    ...
    "my_method": my_method,
}
```

4. 新增算法配置：

```text
testbench/configs/algorithms/my_method.yaml
```

5. 新增或复制 experiment 配置，切换 `algorithm_config`。

统一输出类型定义在：

```text
utils/types.py
```

## 常见问题

### Hugging Face 下载慢或卡住

使用镜像和禁用 Xet：

```bash
env PYTHONUNBUFFERED=1 \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DISABLE_XET=1 \
  .env/bin/python da3/run_da3_batches.py
```

### CUDA 不可用

DA3 Nested 模型需要 CUDA。检查：

```bash
.env/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

如果在沙箱环境中 CUDA 不可见，但宿主机可见，需要在具备 GPU 权限的 shell 中运行 DA3 推理。

### 数值通过但视觉不对

这是可能的。重复平面、局部高重叠、ghosting 和尺度漂移都会制造假阳性。

处理建议：

- 打开 `overlay_top1.ply` 做视觉检查。
- 查看 `top_candidates.md` 中的其他候选。
- 调整 `topview_yaw_step_deg`、scale range、vote distance、trimmed threshold。
- 必要时保留多个子图，不要过早融合进累计 world。

### `.gitingnore` 文件

当前仓库中实际存在的是 `.gitignore`。如果 IDE 中打开了 `.gitingnore`，它可能是误拼写的旧标签。

## 当前不变量摘要

- 所有 Python 依赖安装在 `.env/`。
- DA3 输入图像在 `da3/data/raw/image/`。
- DA3 输出点云和预测在 `da3/data/raw/pointcloud/`。
- 点云配准实验只读取 `da3/data/raw/pointcloud/`，不写入该目录。
- 通用工具放在 `utils/`。
- 算法放在 `algorithm/`。
- 实验入口和配置放在 `testbench/`。
- 实验输出放在 `result/`。
- `.ply`、`.npz`、`result/`、`.env/` 都是派生产物或本地环境，不纳入版本控制。

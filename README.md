# PCR: Point Cloud Registration Benchmark

PCR 是一个 agent 友好的点云刚性配准项目骨架，用于 Unitree + RealSense 室内场景的一对一、多对一、多对多配准实验。

## 实验计划
- 第一阶段：建立统一项目结构、便携环境、任务配置、结果目录和算法注册机制。
- 第二阶段：以 `ransac_only` 作为第一轮基线算法，验证 source -> target 矩阵方向、结果保存和指标报告。
- 第三阶段：接入 `gicp_refine`，使用多次 FPFH+RANSAC coarse transform 作为 small_gicp GICP refine 初值，作为当前推荐主方案。
- 第四阶段：扩展多对一、多对多 workflow，支持 top-k 匹配、绑定结果和游离目标标记。
- 第五阶段：增加指定算法实验、稳定性测试、参数 sweep 和人工结果解释文档。

## 矩阵方向
所有算法和结果必须使用统一方向：

```text
P_target ≈ T @ P_source
```

即 `source` 为待配准点云，`target` 为基准地图点云。

## 目录说明
- `.env/`：便携 Python 环境，推荐通过 conda 创建。
- `dependencies/`：本地 wheel、源码包或离线依赖缓存。
- `configs/`：默认参数和算法示例参数。
- `data/raw/`：原始点云，只读；当前包含空 PLY 占位文件，请替换为真实数据。
- `data/cache/`：预处理缓存。
- `data/tasks/`：pairwise、多对一、多对多任务配置。
- `algorithms/`：只放 pairwise registration 算法实现。
- `workflows/`：组织任务场景，拆解为 pairwise 算法调用。
- `testbench/`：标准测试与指定实验。
- `utils/`：IO、预处理、缓存、指标、可视化、报告和矩阵工具。
- `results/`：每次运行的 cloud、matrix、report、metrics 输出。
- `docs/`：当前任务、算法笔记和结果解释。

## 推荐环境
本机 conda 入口示例：

```bat
%WINDIR%\System32\cmd.exe "/K" D:\coding\Anaconda\Scripts\activate.bat D:\coding\Anaconda
```

创建便携环境：

```bat
D:\coding\Anaconda\Scripts\conda.exe create -p .env python=3.10 -y
.env\python.exe scripts\check_env.py --install
```

仅检测：

```bat
.env\python.exe scripts\check_env.py
```

如需离线安装，将 wheel 文件放入 `dependencies/` 后执行：

```bat
.env\python.exe scripts\check_env.py --install --offline
```

## 依赖
必需依赖：
- `numpy`：矩阵、指标、基础数组运算。
- `PyYAML`：读取配置和任务 YAML。

推荐依赖：
- `open3d`：点云 IO、下采样、normal、FPFH、RANSAC 配准、可视化。`ransac_only` 实际运行需要它；缺失时算法返回 `skipped`。
- `small-gicp`：`gicp_refine` 的 GICP refine 后端。`gicp_refine` 是当前推荐主方案；缺失时算法返回 `skipped`，并在 error 中写明安装命令。

标准库依赖：
- `argparse`、`csv`、`json`、`hashlib`、`pathlib`、`dataclasses`、`time` 等。

第三方许可证：
- NumPy：BSD-3-Clause。
- PyYAML：MIT。
- Open3D：MIT。
- small-gicp：MIT。

## 算法
- `ransac_only`：Open3D FPFH + RANSAC baseline，速度快、结构简单，适合作为粗配准和框架 smoke test。
- `gicp_refine`：当前推荐主方案。默认执行 5 次 FPFH + RANSAC，选择最佳 coarse transform 后调用 small_gicp GICP refine，输出 refined source -> target 变换。

当前推荐 `gicp_refine` 参数：

```yaml
voxel_size: 0.8
distance_threshold_factor: 2.0
ransac_trials: 5
gicp_downsampling_resolution: 0.8
gicp_max_iterations: 20
gicp_max_correspondence_distance_factor: 2.0
```

该配置写入 `configs/gicp_refine.yaml`。specified 实验显示它在当前任务上比 `ransac_only` 和 single/best-of-3 变体更稳定，且精度更高。

`gicp_refine` 关键算法内部指标：
- `algorithm_ransac_trials`
- `algorithm_best_trial`
- `algorithm_best_coarse_fitness`
- `algorithm_best_coarse_inlier_rmse`
- `algorithm_all_trials`
- `algorithm_gicp_converged`
- `algorithm_gicp_error`
- `algorithm_gicp_iterations`
- `algorithm_gicp_num_inliers`
- `algorithm_coarse_time`
- `algorithm_refine_time`

## gicp_refine Specified Experiments
可直接运行：

```bat
.env\python.exe testbench\specified\gicp_refine\ablation.py --config configs\default.yaml --task data\tasks\pair_mission1.yaml
.env\python.exe testbench\specified\gicp_refine\random_stability.py --config configs\default.yaml --task data\tasks\pair_mission1.yaml --repeats 3
.env\python.exe testbench\specified\gicp_refine\parameter_robustness.py --config configs\default.yaml --task data\tasks\pair_mission1.yaml
```

输出位置：
- `results/specified/gicp_refine/ablation/<run_id>/`
- `results/specified/gicp_refine/random_stability/<run_id>/`
- `results/specified/gicp_refine/parameter_robustness/<run_id>/`

每个 run 包含 `metrics.csv`、`metrics.json`、`summary.md`、`report/summary.md`，成功结果还保存 matrix/cloud/overlay。

## 常用命令
推荐使用根目录 bat 入口：

```bat
check_env.bat
run_pairwise.bat --task data\tasks\pair_mission1.yaml
run_multi_to_one.bat --task data\tasks\multi_query_mission1.yaml
run_multi_to_multi.bat --task data\tasks\block_matching_mission1.yaml
run_standard.bat --config configs\default.yaml
```

一对一任务：

```bat
.env\python.exe scripts\run_pairwise.py --task data\tasks\pair_mission1.yaml
```

多对一任务：

```bat
.env\python.exe scripts\run_multi_to_one.py --task data\tasks\multi_query_mission1.yaml
```

多对多任务：

```bat
.env\python.exe scripts\run_multi_to_multi.py --task data\tasks\block_matching_mission1.yaml
```

标准测试：

```bat
.env\python.exe scripts\run_standard.py --config configs\default.yaml
```

## 输出约定
每次运行生成独立 `run_id`，并保存：
- `cloud/`：变换后的 source 点云和 overlay 点云。
- `matrix/`：4x4 txt 矩阵。
- `report/`：Markdown 报告。
- `metrics.csv`：扁平指标表。
- `metrics.json`：完整结构化指标。

一次 standard benchmark 只创建一个 `results/standard/<run_id>/`，所有算法结果写入同一目录。只有 `success` 且 `has_valid_transform=true` 会保存 matrix、cloud 和 overlay；`failed` / `skipped` 的 transformation 为 `None`，只写 metrics、report 和 error。

指标字段拆分：
- `algorithm_*`：算法内部产生的指标，如 `algorithm_fitness`、`algorithm_inlier_rmse`、`algorithm_correspondence_set_size`。
- `eval_*`：workflow/testbench 使用统一 evaluator 重新计算的公共指标，如 `eval_fitness`、`eval_inlier_rmse`、`eval_median_nn_dist`、`eval_trimmed_mean_nn_dist`、`eval_overlap_ratio`、`eval_det_R`、`eval_orthogonality_error`、`eval_translation_norm`。

运行时间字段：
- `preprocess_time`：读取点云、缓存命中/重算、下采样、normal、FPFH 的耗时。
- `algorithm_time`：算法本体耗时。
- `evaluation_time`：统一 evaluator 重新计算公共指标的耗时。
- `artifact_time`：保存 matrix/cloud/overlay 的耗时。
- `total_time`：本条 pairwise 任务总耗时。

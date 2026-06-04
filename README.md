# PCR: Point Cloud Registration Benchmark

PCR 是一个 agent 友好的点云配准项目骨架，用于 Unitree GO2 + 深度点云室内场景的一对一、多对一、多对多配准实验。项目仍以 source -> target 矩阵方向为硬约束；在 DA3 点云存在尺度漂移、重影和相似平面时，允许 specified pipeline 输出带尺度补偿的几何校正候选，但必须明确标记它不等价于严格机器人 SE(3) 位姿。

## 实验计划
- 第一阶段：建立统一项目结构、便携环境、任务配置、结果目录和算法注册机制。
- 第二阶段：以 `ransac_only` 作为第一轮基线算法，验证 source -> target 矩阵方向、结果保存和指标报告。
- 第三阶段：接入 `gicp_refine`，使用多次 FPFH+RANSAC coarse transform 作为 small_gicp GICP refine 初值，作为尺度一致数据的推荐方案。
- 第四阶段：扩展多对一、多对多 workflow，支持 top-k 匹配、绑定结果和游离目标标记。
- 第五阶段：增加指定算法实验、稳定性测试、参数 sweep 和人工结果解释文档。
- 第六阶段：针对 DA3 当前数据，将 `fast_rotation_cluster_scale_vote` 提升为融合前纯点云主方案，用重力约束 yaw、top-view 相关、尺度/平移投票替代易被重影吸偏的自由 GICP refine。

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
- `open3d`：点云 IO、下采样、normal、FPFH、RANSAC 配准、top-view 方案中的点云处理和可视化。`ransac_only`、`gicp_refine`、`multi_comb` 及当前主方案都需要它。
- `small-gicp`：`gicp_refine` 的 GICP refine 后端。`gicp_refine` 用于尺度一致数据；缺失时算法返回 `skipped`，并在 error 中写明安装命令。

标准库依赖：
- `argparse`、`csv`、`json`、`hashlib`、`pathlib`、`dataclasses`、`time` 等。

第三方许可证：
- NumPy：BSD-3-Clause。
- PyYAML：MIT。
- Open3D：MIT。
- small-gicp：MIT。

## 算法
- `ransac_only`：Open3D FPFH + RANSAC baseline，速度快、结构简单，适合作为粗配准和框架 smoke test。
- `gicp_refine`：尺度一致数据的推荐刚体方案。默认执行 5 次 FPFH + RANSAC，选择最佳 coarse transform 后调用 small_gicp GICP refine，输出 refined source -> target 变换。
- `multi_comb`：试验性算法，用于尺度不一致、局部畸变、地板曲面、远端上翘等极不理想数据的诊断。它会生成多个 coarse candidate，用 weighted score 择优，并可选 affine refine。

`multi_comb` 当前支持融合前鲁棒配准诊断：
- 多 coarse source：RANSAC、FGR。
- 多参数扰动：voxel、distance factor、feature subset。
- feature subset：all、non_floor、near_mid_only；high_curvature 已实现但默认不启用，避免百万级点云运行过慢。
- SE(3) 去相关：按旋转差和平移差筛选 diverse top-k candidates。
- top-k rigid refine：默认对 diverse top-3 使用 Open3D GICP refine。
- weighted score：综合公共 evaluator、coverage、plane degeneracy、motion prior。
- GO2/DA3 场景假设：机器人坐标系中 `-Y` 为高度方向，地面通常接近最大 Y。

代码内默认值采用轻量 balanced profile，便于 standard benchmark 回归；完整鲁棒参数组合写在 `configs/multi_comb.yaml`，用于 specified/离线实验。

`multi_comb` 的 affine 模式说明：
- `affine_mode: none`：只输出最佳刚体 coarse transform。
- `affine_mode: constrained`：允许 `R @ diag(sx, sy, sz) @ p_source + t`，只引入三个轴向 scale，不允许 shear。
- `affine_mode: unconstrained`：允许一般 affine `A @ p_source + b`，仅用于几何拟合诊断，不能直接视为机器人刚体位姿。

`multi_comb` 成功输出仍保持 source -> target 方向；但 affine 模式输出不是严格刚体位姿，报告中的 `algorithm_transform_type` 会明确标记。

## 当前主方案：fast_rotation_cluster_scale_vote

当前数据来自 Depth Anything 3 点云，存在点数不一致、重影、尺度漂移、相似平面、远端上翘和曲面地板。实测中，自由 RANSAC/GICP 很容易被局部高 fitness 的错误重叠区域吸偏；视觉上正确的候选反而常常不是传统指标第一名。因此当前主方案不是单一 `algorithms/` 注册算法，而是 specified pipeline：

```text
top-view yaw/scale/translation coarse candidates
-> rotation cluster
-> per-cluster representative
-> horizontal/vertical scale + translation vote
-> bounded/no local refine
-> visual check and fusion gate
```

入口：

```bat
.env\python.exe testbench\specified\multi_comb\fast_rotation_cluster_scale_vote.py --config configs\ransac_all_no_refine.yaml
```

核心假设：
- 机器人坐标系中 `-Y` 为高度方向。
- source 和 target 之间主要是 yaw、水平平移、有限高度偏移和 DA3 造成的尺度差异。
- 当前阶段只做融合前配准候选，不更新地图，不做 pose graph，不把尺度补偿矩阵当作严格里程计。

算法流程：
- 预处理：读取 `data/raw/16/incremental.ply` 和 `data/raw/14/world.ply`，使用现有 cache/preprocessing 工具得到下采样点云。
- top-view 候选：将点云投影到 XZ 平面，枚举 yaw 和水平尺度，用 2D 占据相关寻找水平平移峰值。
- 候选保留：默认保留 top-view 前 200 个候选，避免正确方向族因局部重影分数偏低而被过早丢弃。
- 旋转聚类：按旋转角差聚类，默认阈值 5 度。
- 精筛代表：默认取前 27 个 rotation cluster 的代表进入尺度/平移投票，同时保留少量 top score 候选。
- 尺度投票：在固定 yaw 方向附近，搜索 `R @ diag(sx, sy, sx)`，允许水平尺度和高度尺度不同。
- 平移投票：对采样点的最近邻平移向量做 voxel voting，用中位数估计稳定平移。
- refine 策略：默认关闭 GICP；只有在后续明确需要时才启用受限 GICP，避免重影和尺度错判把候选吸偏。
- 排名：以 refine 后公共评估分数为主，而不是单纯 cluster 大小或 top-view peak。

默认落地参数：

```text
candidate_mode = topview
topview_candidates = 200
top_clusters = 27
top_score_candidates = 16
max_refine_candidates = 27
cluster_rotation_deg = 5.0
horizontal_scale_range = 0.75 .. 1.20, steps=5
vertical_scale_range = 0.70 .. 1.15, steps=4
max_vote_points = 240
gicp_max_iterations = 0
```

验收记录：
- run：`results/specified/multi_comb/fast_rotation_cluster_scale_vote/fast_rotation_cluster_scale_vote_20260604_170042`
- 内部 pipeline 总耗时：`5.770699s`
- top1 落在人工确认的正确旋转族。
- top3 中有两个正确旋转族候选。
- 视觉检查显示方案基本正确，主要残余误差集中在高度方向，约 0.3m。

结果解释：
- `eval_det_R != 1` 或 `eval_orthogonality_error` 较大时，矩阵包含尺度补偿，不是严格 SE(3) 位姿。
- `scale_values.sx/sz` 反映水平尺度补偿，`scale_values.sy` 反映高度尺度补偿。
- 对当前数据，`eval_fitness` 很高仍可能是假阳性，必须结合 overlay 视觉检查、rotation family 和尺度合理性判断。
- 当前推荐将该矩阵作为融合前几何校正候选；真正地图融合阶段仍需要 quality gate、pose graph 或 submap 策略，避免逐步贪心更新累积误差。

`gicp_refine` 推荐刚体参数：

```yaml
voxel_size: 0.8
distance_threshold_factor: 2.0
ransac_trials: 5
gicp_downsampling_resolution: 0.8
gicp_max_iterations: 20
gicp_max_correspondence_distance_factor: 2.0
```

该配置写入 `configs/gicp_refine.yaml`。specified 实验显示它在尺度一致/无明显重影任务上比 `ransac_only` 和 single/best-of-3 变体更稳定，且精度更高；但在当前 DA3 尺度漂移数据上，主方案以 `fast_rotation_cluster_scale_vote` 为准。

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

## multi_comb Specified Experiment
批量一对一参数测试：

```bat
.env\python.exe testbench\specified\multi_comb\pair_batch_parameter_test.py --task data\tasks\pair_batch_incremental_world.yaml --config configs\multi_comb.yaml
.env\python.exe testbench\specified\multi_comb\world16_to_world14_parameter_sweep.py --config configs\multi_comb.yaml
.env\python.exe testbench\specified\multi_comb\fast_rotation_cluster_scale_vote.py --config configs\ransac_all_no_refine.yaml
```

输出位置：
- `results/specified/multi_comb/pair_batch_parameter_test/<run_id>/`
- `results/specified/multi_comb/world16_to_world14_parameter_sweep/<run_id>/`

输出包含 `metrics.csv`、`metrics.json`、`summary.md`、`report/summary.md`、`report/visualization_commands.md`，成功结果保存 matrix/cloud/overlay。

当前未实现地图融合；建议后续在独立分支/模块中加入 quality gate、pose graph、submap fusion，避免逐步贪心融合放大误差。

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

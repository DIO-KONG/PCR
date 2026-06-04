# Algorithm Notes

## ransac_only
- 依赖 Open3D。
- 输入为已经完成下采样、normal、FPFH 的 source/target。
- 使用 FPFH 特征匹配和 RANSAC 输出 source -> target 矩阵。
- 缺失 Open3D 时返回 `skipped`，不伪造成功。
- 零 correspondence 或零 fitness 视为 `failed`，不保存 matrix/cloud/overlay。
- 当前示例默认使用 coarse baseline：`voxel_size=0.8`，`distance_threshold_factor=2.0`。

## gicp_refine
- 依赖 Open3D 和 small-gicp。
- 输入使用 workflow 已准备好的 `pcd_down`、normal 和 FPFH。
- 默认重复执行 5 次 FPFH + RANSAC，从 coarse candidates 中选择 fitness 最高、RMSE 最低的 transform。
- 使用 small_gicp GICP 对最佳 coarse transform refine。
- 输出仍为 source -> target。
- 记录 `algorithm_coarse_time`、`algorithm_refine_time`，总算法耗时写入 `algorithm_time`。
- 当前推荐参数来自 specified 实验：`voxel_size=0.8`、`distance_threshold_factor=2.0`、`ransac_trials=5`、`gicp_downsampling_resolution=0.8`。
- 经验结论：best-of-5 GICP 比 single/best-of-3 更稳定；`coarse_voxel_size=0.8` 和 `gicp_voxel_size=0.8` 在当前数据上明显优于 0.6/1.0。
- 适用边界：当 source/target 尺度一致且 coarse transform 已接近正确 basin 时表现好；当 DA3 点云存在明显尺度漂移、重影和相似平面时，GICP 可能被局部错误重叠吸偏。

## multi_comb
- 试验性算法，用于诊断尺度、局部畸变、曲面地板、远端上翘等非理想数据。
- coarse candidate 支持 Open3D RANSAC；FGR 在 Open3D API 可用时支持，否则单个 candidate 记录 failed。
- weighted score 同时考虑 eval_fitness、eval_overlap_ratio、eval_inlier_rmse、eval_trimmed_mean_nn_dist 和 eval_translation_norm。
- affine_mode 为 constrained/unconstrained 时输出不再是严格刚体位姿，仅用于几何拟合诊断。
- 当前增强版还会计算 coverage_score、plane_degeneracy、motion_prior_error，并用这些指标辅助候选排序。
- 默认生成多来源、多参数、多点子集 coarse candidates，并通过 SE(3) rotation/translation 去相关筛选 top-k refine candidates。
- `non_floor` 利用机器人坐标约定：`-Y` 是高度，地面/低矮结构通常接近最大 Y。
- `high_curvature` 子集已实现但默认不启用，适合后续离线实验。

## fast_rotation_cluster_scale_vote
- 指定实验脚本：`testbench/specified/multi_comb/fast_rotation_cluster_scale_vote.py`。
- 当前主方案，用于 10 秒内纯点云融合前配准候选筛选，不新增 algorithms 统一接口。
- 场景假设：Unitree GO2 机器人坐标系中 `-Y` 为高度方向；DA3 点云存在尺度漂移、重影、点数不一致、远端上翘和相似平面。
- 设计目标：先稳定找对方向族，再补偿尺度和平移；不追求把输出解释为严格 SE(3) 里程计。

### 输入与输出
- 输入：`data/raw/16/incremental.ply` 作为 source，`data/raw/14/world.ply` 作为 target。
- 输出目录：`results/specified/multi_comb/fast_rotation_cluster_scale_vote/<run_id>/`。
- 输出矩阵方向仍为 source -> target。
- 输出矩阵允许包含 `R @ diag(sx, sy, sz)`，其中 `sx≈sz` 是水平尺度补偿，`sy` 是高度尺度补偿。
- 当 `eval_det_R != 1` 或 `eval_orthogonality_error` 较大时，结果不是刚体位姿。

### 算法流程
- 预处理：复用 `utils.preprocessing.prepare_point_cloud_from_path_with_cache()`，读取点云并使用缓存。
- Top-view 投影：将 source/target 投影到 XZ 平面，生成 2D occupancy grid。
- Yaw/scale 枚举：默认枚举全角度 yaw 和水平尺度，利用 2D FFT 相关寻找水平平移峰值。
- 候选保留：默认保留 top-view 分数前 200 个候选，避免正确方向族被局部重影压低后提前丢弃。
- Rotation clustering：按旋转角差聚类，默认阈值 `5deg`。
- Cluster representative：默认取前 27 个 rotation cluster 的代表进入精筛。
- Scale + translation vote：在候选 yaw 附近搜索 `R @ diag(sx, sy, sx)`，用最近邻平移向量做 voxel voting，并用中位数估计平移。
- Ranking：以 refine 后公共指标和投票稳定性为主，避免单纯按 cluster size 或 top-view peak 选出错误结构。
- Artifact：保存 top cluster 的 matrix、registered cloud、overlay 和可视化命令。

### 默认参数
- `candidate_mode=topview`
- `topview_candidates=200`
- `top_clusters=27`
- `top_score_candidates=16`
- `max_refine_candidates=27`
- `cluster_rotation_deg=5.0`
- `horizontal_scale_min=0.75`
- `horizontal_scale_max=1.20`
- `horizontal_scale_steps=5`
- `vertical_scale_min=0.70`
- `vertical_scale_max=1.15`
- `vertical_scale_steps=4`
- `max_vote_points=240`
- `gicp_max_iterations=0`

### 评分与人工检查
- `eval_fitness` 在当前数据上容易被重影和相似平面抬高，不能单独作为成功依据。
- 推荐同时检查 rotation family、`eval_trimmed_mean_nn_dist`、`scale_values`、overlay 视觉结果和高度残差。
- 正确候选通常表现为方向族稳定、水平尺度合理、重叠区域形状一致；错误候选可能 fitness 高但整体朝向或地图区域错。

### 验收记录
- 2026-06-04 默认 run：`fast_rotation_cluster_scale_vote_20260604_170042`。
- 内部总耗时：`5.771s`。
- top1 落在人工确认的正确旋转族。
- top3 中有两个正确旋转族候选。
- 人工 overlay 检查：整体基本正确，主要残余误差在高度方向，约 `0.3m`。

### 后续建议
- 若高度仍偏差约 0.3m，优先尝试高度方向的鲁棒平移再估计，而不是放开 GICP 自由 refine。
- 融合阶段必须加入 quality gate；不建议把每一步输出直接贪心写回地图。
- 有机器人可用后，应接入 IMU/重力和里程计作为 soft prior，用 pose graph 或 submap 管理累积误差。

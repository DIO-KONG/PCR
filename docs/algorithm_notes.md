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

## multi_comb
- 试验性算法，用于诊断尺度、局部畸变、曲面地板、远端上翘等非理想数据。
- coarse candidate 支持 Open3D RANSAC；FGR 在 Open3D API 可用时支持，否则单个 candidate 记录 failed。
- weighted score 同时考虑 eval_fitness、eval_overlap_ratio、eval_inlier_rmse、eval_trimmed_mean_nn_dist 和 eval_translation_norm。
- affine_mode 为 constrained/unconstrained 时输出不再是严格刚体位姿，仅用于几何拟合诊断。
- 当前增强版还会计算 coverage_score、plane_degeneracy、motion_prior_error，并用这些指标辅助候选排序。
- 默认生成多来源、多参数、多点子集 coarse candidates，并通过 SE(3) rotation/translation 去相关筛选 top-k refine candidates。
- `non_floor` 利用机器人坐标约定：`-Y` 是高度，地面/低矮结构通常接近最大 Y。
- `high_curvature` 子集已实现但默认不启用，适合后续离线实验。

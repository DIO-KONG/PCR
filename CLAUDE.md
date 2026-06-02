# PCR 项目全局 CLAUDE 说明

## 工程目标
- 本项目用于点云刚性配准研究与 benchmark，适配 Unitree + RealSense 室内场景。
- 所有矩阵输出必须统一方向：P_target ≈ T @ P_source。
- 项目旨在支持算法注册、标准化测试、不同场景 workflow 以及 agent 协作总结。

## 环境约束
- 使用便携 Python 环境 `.env/`。
- 不依赖系统 PATH 中的 python、conda 或 pip。
- 可在 `dependencies/` 下本地化安装缺失依赖。
- 所有第三方依赖必须在 README.md 中明确说明其用途和是否必需。
- 根目录 `.bat` 入口必须调用 `.env\python.exe`，不得依赖 PATH。

## 文件夹职责

### algorithms/
- 放置算法实现文件。
- 算法必须遵循统一接口：输入预处理点云、参数字典；输出 RegistrationResult。
- 禁止在算法中实现实验流程、可视化、报告生成或扫描多算法。
- 公共小函数仅限算法内部辅助，例如矩阵方向校验、错误封装、结果对象构造。

### workflows/
- 管理不同配准场景 (一对一、多对一、多对多) 的执行逻辑。
- 拆解复杂任务为 pairwise 调用，并处理绑定、top-k、游离标记。
- 不直接实现算法，只调算法接口和 utils。
- multi_to_one 默认返回每个 source 的结果；只有显式 `mode: top_k` 或 `apply_top_k: true` 才做全局 top-k 截断。

### testbench/
- 包含实验逻辑。
- standard/：通用标准测试。
- specified/：算法或 pipeline 的针对性测试。
- 不应由算法调用 testbench 模块。

### utils/
- 提供共享工具：IO、预处理、缓存、评估、矩阵工具、可视化、报告。
- 算法或 workflow 调用 utils，但 utils 不负责执行实验或算法逻辑。
- 预处理入口优先使用 `prepare_point_cloud_from_path_with_cache()`，缓存键必须包含源文件签名和预处理配置。
- 公共评估指标输出字段统一为 eval_fitness、eval_inlier_rmse、eval_median_nn_dist、eval_trimmed_mean_nn_dist、eval_overlap_ratio、eval_det_R、eval_orthogonality_error、eval_translation_norm。

### data/
- raw/：只读原始点云。
- cache/：预处理缓存。
- tasks/：任务配置 (YAML/JSON) 描述实验或 workflow 场景。

### results/
- 保存实验结果。
- 分为 standard/ 和 specified/，每次运行生成独立 `<run_id>` 目录。
- 不允许在结果目录放置源代码。
- 一次 standard benchmark 只能创建一个 `results/standard/<run_id>/`，所有算法结果写入同一 run 目录。

## 算法接口约束
- 输入：预处理后的 source 和 target 点云，参数字典。
- 输出：
  - method: 算法名
  - status: success / failed / skipped
  - success 时输出 4x4 transformation matrix
  - failed / skipped 时 transformation 必须为 None，`has_valid_transform=false`
  - algorithm_time, 参数使用情况, 错误信息
- 不允许伪造成功状态；缺失依赖返回 skipped。
- 矩阵方向必须符合 source -> target；方向不明确时需同时评估 T 和 T^-1。
- 算法内部指标必须使用 `algorithm_*` 字段记录；公共评估指标必须使用 `eval_*` 字段记录。

## 可视化约束
- 允许 GUI 弹窗展示点云。
- 默认保存 transformed cloud 到结果目录。
- overlay 默认：target=灰色, transformed source=黄色。

## 结果约束
- 只有 `success` 结果保存矩阵、cloud 和 overlay；`failed` / `skipped` 只写 metrics、report 和 error。
- 保存矩阵为 4x4 txt，格式统一。
- CSV/JSON/Markdown 报告需包含算法、参数、运行指标、矩阵路径、cloud 路径、状态/错误信息。
- 每条结果必须记录 preprocess_time、algorithm_time、evaluation_time、artifact_time、total_time。
- 每次运行生成独立目录，避免覆盖历史数据。

# pcr 核心框架

`pcr/` 是项目长期维护的核心包。旧的 `testbench/` 脚本保留为实验入口和兼容 CLI，但主要实现不再继续堆在 runner 里。

## 模块边界

```text
pcr/domain/        稳定数据模型与 Transform 方向约定
pcr/config/        YAML 读取和配置到任务模型的转换
pcr/io/            DA3 NPZ 与 PLY 的基础读写
pcr/preprocessing/ 缺失预处理结果的补齐入口
pcr/algorithms/    shared-frame 粗配准和 bounded ICP 等数值算法
pcr/evaluation/    评分与指标计算
pcr/state/         WorldState、transform graph 和融合策略
pcr/artifacts/     PLY、matrix、JSON、Markdown 的统一写入
pcr/pipeline/      单步配准和完整 sequence 的流程编排
pcr/app/           CLI 顶层应用
```

依赖方向保持单向：

```text
app -> config / pipeline / artifacts
pipeline -> algorithms / state / domain
algorithms -> domain / evaluation / io
state -> domain / fusion
artifacts -> domain / io
domain -> numpy/dataclass
```

算法模块不能写 `result/`；artifact 模块不能知道 RANSAC/ICP 的内部公式；runner 不能直接拼 JSON 或保存矩阵。

## Transform 约定

全项目统一：

```text
p_target = T_source_to_target @ p_source
```

使用 `pcr.domain.Transform` 保存矩阵时必须带上 `source` 和 `target`。例如：

```python
T_source_to_target = Transform("batch_02", "batch_01", matrix)
T_target_to_global = Transform("batch_01", "batch_01", np.eye(4))
T_source_to_global = T_source_to_target.then(T_target_to_global)
```

`then()` 会检查前一个变换的 target 是否等于后一个变换的 source，避免把矩阵方向写反。

## 当前 walk-forward 数据流

1. `pcr.app.run_walkforward` 读取 YAML，创建 `ArtifactStore`。
2. `pcr.config.loader` 把 YAML 转成 `SequenceTask`。
3. `pcr.pipeline.SequencePipeline` 初始化 `WorldState`。
4. `RegistrationStepPipeline` 对每个 step 执行：
   - 补齐 floor-removed 点云；
   - 动态共享帧 4选3；
   - RANSAC + Kabsch 粗配准；
   - 组合 source 到 global 的初值；
   - bounded point-to-plane ICP；
   - 返回 `StepResult`。
5. `WorldState.apply_step()` 用 fusion policy 更新累计 world。
6. `ArtifactStore` 写每步 overlay、matrix、metrics 和最终 summary。

当前 fusion policy 是 `AppendVoxelFusion`，语义与旧 runner 一致：`world + registered_source` 后做 voxel downsample。后续重影过滤应作为新的 fusion policy 实现。

## 兼容入口

旧命令仍然可用：

```bash
.env/bin/python testbench/run_shared_frame_walkforward.py \
  --config testbench/configs/experiments/shared_frame_walkforward.yaml \
  --run-name dynamic_top3_bounded_icp_walkforward \
  --overwrite
```

该脚本现在只转调：

```text
pcr.app.run_walkforward
```

Pairwise 和参数 sweep 也遵循同样规则：

```text
testbench/run_registration.py -> pcr.app.run_registration
testbench/sweep_shared_frame_alignment.py -> pcr.app.sweep_shared_frame_alignment
```

三类入口都复用 `pcr.algorithms.shared_frame` 中的 DA3 NPZ 读取、共享帧 correspondence、Kabsch/RANSAC/refit、candidate scoring，不再保留旧的重复数值实现。

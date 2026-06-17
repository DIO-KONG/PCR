# testbench

本目录放置实验配置和实验入口。

运行当前 DA3 pairwise 多方案比较：

```bash
.env/bin/python testbench/run_registration.py --run-name current_da3_pairwise --overwrite
```

配置结构：

- `configs/datasets/`：数据集 source/target 配置。
- `configs/algorithms/`：单个算法参数。
- `configs/experiments/`：组合多个算法方案的实验配置。

实验输出默认写入 `result/registration_schemes/`，该目录由 `.gitignore` 忽略。

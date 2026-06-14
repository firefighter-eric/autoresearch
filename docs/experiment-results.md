# 实验结果记录规范

这个仓库把本机原始产物和可提交的对比摘要分开管理。

## 当前 RTX 5080 结论

当前可提交的最佳配置来自 `jun14-rtx5080-cuda5080` 这轮实验：

```text
UNEMBEDDING_LR = 0.006
WARMDOWN_RATIO = 0.75
FINAL_LR_FRAC = 0.05
```

相对 `cuda-5080` baseline：

```text
baseline val_bpb: 1.124884
best val_bpb:     1.118723
improvement:      0.006161，约 0.55%
```

本轮实验结果：

```text
status   val_bpb   change
keep     1.124008  WARMDOWN_RATIO 0.5 -> 0.75
keep     1.121219  FINAL_LR_FRAC 0.0 -> 0.05
keep     1.118723  UNEMBEDDING_LR 0.004 -> 0.006
discard  1.127275  EMBEDDING_LR 0.6 -> 0.8
discard  1.174248  cuda-5080 depth 8 -> 9
discard  1.123302  RoPE base 10k -> 200k
```

主要 insight：

1. **这台 5080 更吃训练吞吐，不适合先加大模型。** `depth 8 -> 9` 把参数从 `50.3M` 提到 `80.9M`，但 5 分钟内训练量从 `97.5M/100M` tokens 降到 `62.4M` tokens，`val_bpb` 明显变差到 `1.174248`。在这个固定 5 分钟预算下，容量增加没有抵消吞吐下降。
2. **LR schedule 是当前最有效的低风险方向。** `WARMDOWN_RATIO 0.75` 和 `FINAL_LR_FRAC 0.05` 都不改变模型大小和显存，仍保持约 `100M` tokens 训练量，但连续降低 `val_bpb`。这说明 5080 profile 的默认学习率衰减可能太晚、末尾归零也偏保守。
3. **lm_head 学习率还有上调空间。** `UNEMBEDDING_LR 0.004 -> 0.006` 在已有 schedule 改动基础上继续提升到 `1.118723`。这说明输出层在当前时间预算下可能欠更新，适合继续小步 sweep，例如 `0.005`、`0.007`、`0.008`。
4. **embedding LR 上调不一定迁移。** H100 历史里 `embedding LR 0.6 -> 0.8` 有效，但在当前 RTX 5080 baseline 上变差到 `1.127275`。原因可能是 5080 profile 的 batch、步数、吞吐和 schedule 不同，embedding 更高 LR 在较少训练量下更容易过冲。
5. **RoPE base 200k 在当前 2K context 和配置下无收益。** 它在 H100 历史实验里有效，但本轮在已改 schedule 后变差到 `1.123302`。当前 seq len 是 `2048`，不应把更大 RoPE base 当作默认必需项。

下一轮优先方向：

1. 围绕 `UNEMBEDDING_LR` 做小范围 sweep：`0.005`、`0.007`、`0.008`。
2. 围绕 schedule 做小范围 sweep：`WARMDOWN_RATIO 0.65/0.8`、`FINAL_LR_FRAC 0.03/0.08`。
3. 优先尝试不降低吞吐的优化参数；暂缓加深、加宽或显著增大 attention 计算。
4. 如果尝试模型结构变化，必须同时关注 `total_tokens_M`，不能只看训练 loss。

## 哪些进 Git

提交这些文件：

```text
benchmarks/results-summary.tsv
docs/experiment-results.md
README.md
program.md
pyproject.toml
uv.lock
```

不要提交这些本地产物：

```text
results/
run.log
prepare.log
results.tsv
.venv/
__pycache__/
```

`uv.lock` 保留在 repo 里，但不同 CUDA、MPS、CPU、Python 版本和 package index 设置可能会让 `uv` 重写它。不要提交本机自动产生的 lockfile diff，除非这次改动的目的就是更新依赖。

## 原始结果目录

每次 run 都放在一个被 git ignore 的目录下：

```text
results/<run-tag>/
```

`run-tag` 建议包含日期、设备和 profile：

```text
jun14-rtx5080-cuda5080
jun14-mac-mps
jun14-mac-mps-bf16
jun14-h100-cuda-large
```

推荐文件结构：

```text
results/<run-tag>/results.tsv
results/<run-tag>/prepare.log
results/<run-tag>/000-baseline-<device>-<profile>.log
results/<run-tag>/<NNN>-<short-commit>-<status>-<slug>.log
```

示例：

```text
results/jun14-rtx5080-cuda5080/000-baseline-rtx5080-cuda5080.log
results/jun14-mac-mps/000-baseline-mac-mps.log
results/jun14-mac-mps-bf16/000-baseline-mac-mps-bf16.log
results/jun14-rtx5080-cuda5080/012-a1b2c3d-keep-rope-base-200k.log
```

路径和文件名使用小写英文。`device` 和 `profile` 名称要稳定，这样不同机器的结果可以直接按目录和文件名对比。

## 单次 Run 的 TSV

每个原始结果目录都有一个本地 `results.tsv`。这个文件不进 git。

表头：

```text
commit	val_bpb	memory_gb	status	description
```

字段含义：

```text
commit       短 git commit hash，baseline 可写 baseline
val_bpb      validation bits per byte，越低越好
memory_gb    峰值显存/内存 GB，保留一位小数
status       keep、discard 或 crash
description  简短实验说明，可以写中文
```

示例：

```text
commit	val_bpb	memory_gb	status	description
baseline	1.124884	6.0	keep	5080 基线，cuda-5080 profile
a1b2c3d	1.118200	6.2	keep	提高 embedding lr
b2c3d4e	1.130000	6.0	discard	RoPE base 200k
c3d4e5f	0.000000	0.0	crash	batch 太大，OOM
```

## 跨机器汇总表

使用 `benchmarks/results-summary.tsv` 做轻量、可提交的跨机器对比。

表头：

```text
date	machine	device	profile	commit	status	val_bpb	memory_gb	num_steps	total_tokens_M	mfu_percent	training_seconds	total_seconds	seq_len	num_params_M	depth	log_ref	description
```

每个值得保留的结果记录一行。这个汇总表只放从日志末尾 summary 复制出来的最终数字，不要把完整日志粘进来。

每条日志需要提取这些值：

```text
val_bpb
peak_vram_mb -> 换算为 memory_gb
num_steps
total_tokens_M
mfu_percent
training_seconds
total_seconds
num_params_M
depth
```

同时记录这些上下文：

```text
date
machine
device
profile
commit
status
seq_len
log_ref
description
```

`log_ref` 指向本机 ignored 的原始日志路径，例如：

```text
results/jun14-rtx5080-cuda5080/000-baseline-rtx5080-cuda5080.log
```

## 工作流

1. 创建 `results/<run-tag>/`。
2. 如果需要准备数据，运行：`uv run prepare.py > results/<run-tag>/prepare.log 2>&1`。
3. 运行训练：`uv run train.py > results/<run-tag>/<log-name>.log 2>&1`。
4. 提取指标：`grep "^val_bpb:\|^peak_vram_mb:\|^num_steps:\|^total_tokens_M:" results/<run-tag>/<log-name>.log`。
5. 把原始结果追加到 `results/<run-tag>/results.tsv`。
6. 把重要的可比较结果追加到 `benchmarks/results-summary.tsv`。

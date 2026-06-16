# 实验结果记录规范

这个仓库把本机原始产物和可提交的对比摘要分开管理。

## 数据集和 Token 口径

当前本机缓存的数据来自 `~/.cache/autoresearch/data`，按 `prepare.py` 里的当前 tokenizer 精确统计如下。训练 dataloader 会给每篇文档 prepend 一个 BOS token，所以训练口径优先看 `with_bos_tokens`。

```text
split  shards  rows     text_tokens  with_bos_tokens
train  10      846848   630394969    631241817
val    1       84992    63289984     63374976
all    11      931840   693684953    694616793
```

当前 10 个 train shard 对 5 分钟 RTX 5080 实验是够的：普通 `cuda-5080` run 每次约 `99.4M` tokens，只覆盖本地 train set 的约 `15.7%`。这些实验是在同一批数据上比较不同参数，累计值应理解为 token-passes，不是不同 token 数。

```text
run type                         steps  tokens
cuda-5080 normal                 379    99352576  (~99.4M)
cuda-5080 device_batch_size=32   397    104071168 (~104.1M)
jun16 20-run sweep total         -      1991770112 (~1.99B token-passes)
```

`program.md` 里的默认示例没有明确机器型号，但 `peak_vram_mb=45060.2`、`mfu_percent=39.80`，基本对应 H100 / 大显存 CUDA 级别机器。它的训练量是：

```text
953 steps * 524288 tokens/step = 499646464 tokens (~499.6M)
```

按 `prepare.py` 的全量配置，数据集最多包含 `6542` 个 train shard 加 `1` 个 pinned val shard。基于当前本地 shard 的平均 token 密度粗估，全量数据约为：

```text
full train estimate:  ~412.4B text tokens, ~413.0B with BOS
full total estimate:  ~412.5B text tokens, ~413.0B with BOS
```

## 当前 Mac MPS 结论

当前可提交的 Mac 改动是对 MPS 开启 `bfloat16` autocast，但参数仍保持 `float32`：

```text
device.type == "mps" -> torch.amp.autocast(device_type="mps", dtype=torch.bfloat16)
get_parameter_dtype("mps") -> torch.float32
```

相对 `mps` fp32 baseline：

```text
baseline val_bpb: 1.572640
bf16 val_bpb:     1.486914
improvement:      0.085726，约 5.45%
```

本轮实验结果：

```text
status   val_bpb   steps   tokens_M   memory_mb   total_s   change
keep     1.572640  157     10.3       356.6       542.8     mps fp32 baseline
keep     1.486914  192     12.6       220.6       480.2     enable MPS bfloat16 autocast
```

主要 insight：

1. **MPS bf16 autocast 是明确正收益。** 固定 5 分钟训练预算下，optimizer steps 从 `157` 增加到 `192`，训练 token 从 `10.3M` 增加到 `12.6M`，`val_bpb` 从 `1.572640` 降到 `1.486914`。
2. **这次收益主要来自吞吐提升。** 每步时间下降后，Mac 在同样训练预算里多做了约 `22%` 的 optimizer step；这和此前 H100/5080 经验一致：小设备上先提高有效训练量，比盲目加大模型更稳。
3. **显存/统一内存占用也下降。** `peak_vram_mb` 从 `356.6` 降到 `220.6`，说明只开 autocast 不但没有增加内存压力，反而降低了高流量算子的临时张量成本。
4. **先保留 fp32 参数更稳。** 当前改动没有把 embedding、optimizer state 或模型参数整体转 bf16，只让 forward/loss 的可 autocast 算子使用 bf16；这是低风险路径，后续如果要试参数 bf16，应单独作为实验记录。

下一轮优先方向：

1. 保留 MPS bf16 autocast 作为 Mac 默认路径。
2. 单独测试 `get_parameter_dtype("mps") -> torch.bfloat16`，确认 embedding/value embedding 转 bf16 是否继续提升，还是带来数值或 optimizer 风险。
3. 继续优先尝试不降低吞吐的 Mac profile 改动，例如 schedule、LR、batch/accumulation，而不是加深或加宽模型。

## 当前 RTX 5080 结论

当前可提交的 RTX 5080 配置来自 `jun16-rtx5080-cuda5080-param20` 这轮 W&B online sweep：

```text
UNEMBEDDING_LR = 0.006
WARMDOWN_RATIO = 0.62
FINAL_LR_FRAC = 0.05
```

历史最佳和本轮线上可比结果要分开看：

```text
original cuda-5080 baseline:        1.124884
historical best before W&B online:  1.118723
W&B online control, same params:    1.119551
W&B online best, warmdown 0.62:     1.118794
online control improvement:         0.000757，约 0.068%
```

`1.118794` 仍比历史 `1.118723` 高 `0.000071`，但它是在 W&B online 逐步 logging 开启后的可比最佳。因为 LR schedule 按 wall-clock progress 衰减，W&B online 的额外开销会轻微改变每一步对应的 LR，因此本轮用 online control 作为主要参照。

本轮 20 次 W&B online 实验结果：

```text
status       val_bpb   W&B run     change
discard      1.119737  cm2qqiow    UNEMBEDDING_LR 0.006 -> 0.007
discard      1.119912  w6wi1pjv    UNEMBEDDING_LR 0.006 -> 0.008
discard      1.120546  4t4woiap    UNEMBEDDING_LR 0.006 -> 0.005
discard      1.120185  rizmhpvk    WARMDOWN_RATIO 0.75 -> 0.80
discard      1.119161  4ifqa6g4    WARMDOWN_RATIO 0.75 -> 0.65
discard      1.119448  vi00chnm    FINAL_LR_FRAC 0.05 -> 0.08
discard      1.120227  xzaesero    FINAL_LR_FRAC 0.05 -> 0.03
discard      1.128194  zr93krib    device_batch_size 16 -> 32
discard      1.119294  8b36y5s0    MATRIX_LR 0.04 -> 0.045
discard      1.120488  6ifzdxrj    WEIGHT_DECAY 0.20 -> 0.15
control      1.119551  34ey2vzp    online rerun of historical best params
keep-online  1.118889  xk59xt0t    WARMDOWN_RATIO 0.75 -> 0.60
discard      1.119393  smirnx9y    WARMDOWN_RATIO 0.60 -> 0.55
discard      1.119085  2wn294ua    WARMDOWN_RATIO 0.60 -> 0.58
keep-online  1.118794  mupl9ffj    WARMDOWN_RATIO 0.60 -> 0.62
discard      1.118904  rtdsgy9i    WARMDOWN_RATIO 0.62 -> 0.63
discard      1.118858  f7imtocw    WARMDOWN_RATIO 0.62 -> 0.61
discard      1.118995  5i37diu3    WARMDOWN_RATIO 0.62 -> 0.625
discard      1.118829  6t5lhxpn    WARMDOWN_RATIO 0.62 plus MATRIX_LR 0.045
discard      1.119033  0ryet5y7    WARMDOWN_RATIO 0.62 plus FINAL_LR_FRAC 0.06
```

主要 insight：

1. **本轮主要收益来自更短 warmdown。** W&B online control 是 `1.119551`，`WARMDOWN_RATIO=0.62` 降到 `1.118794`。固定 5 分钟预算下，online logging 让 schedule 对 wall-clock 更敏感，0.75 会偏早进入低 LR 的后半段；0.62 保留了更长高 LR 训练，同时仍有足够末段收敛。
2. **0.62 附近存在清晰局部最优。** `0.60` 是 `1.118889`，`0.61` 是 `1.118858`，`0.62` 最好，`0.625` 和 `0.63` 又回退。这说明方向不是随机单点，而是一个窄窗口。
3. **lm_head LR 的旧结论没有继续外推。** 在 `UNEMBEDDING_LR=0.006` 基础上改到 `0.005/0.007/0.008` 都变差，说明当前输出层 LR 已经接近合适，不再需要继续加。
4. **更大 microbatch 速度更快但质量明显变差。** `device_batch_size 16 -> 32` 把 tokens 从 `99.4M` 提到 `104.1M`、MFU 从约 `9.01%` 到 `9.44%`，但 `val_bpb` 变差到 `1.128194`，并且显存涨到约 `11.4GB`。这不是值得保留的吞吐优化。
5. **组合调参没有超过单独 warmdown。** `0.62 + MATRIX_LR=0.045` 得到 `1.118829`，非常接近但仍差于 `0.62`；`0.62 + FINAL_LR_FRAC=0.06` 得到 `1.119033`。当前应该先保留 `WARMDOWN_RATIO=0.62`，不要叠加这些组合。

下一轮优先方向：

1. 如果继续 5080 online sweep，优先小步测 `WARMDOWN_RATIO=0.615/0.618/0.622`，不要再大范围扫。
2. 可以单独测试 `MATRIX_LR=0.035` 或 `WEIGHT_DECAY=0.25`，但要以 `WARMDOWN_RATIO=0.62` 为底座。
3. 暂缓 `device_batch_size=32`、加深模型、RoPE base 200k 这类已经显示负收益或与当前 2K context 不匹配的方向。

## 哪些进 Git

提交这些文件：

```text
benchmarks/results-summary.tsv
docs/experiment-results.md
README.md
program.md
pyproject.toml
```

不要提交这些本地产物：

```text
results/
wandb/
run.log
prepare.log
results.tsv
uv.lock
.venv/
__pycache__/
```

`uv.lock` 不进 repo。不同 CUDA、MPS、CPU、Python 版本和 package index 设置会让 `uv` 生成不同 lockfile，本地保留即可。

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
experiment	val_bpb	memory_gb	status	wandb_url	description
```

字段含义：

```text
experiment   稳定 run name，和日志文件名、W&B run name 对齐
val_bpb      validation bits per byte，越低越好
memory_gb    峰值显存/内存 GB，保留一位小数
status       keep、discard 或 crash
wandb_url    如果启用 W&B，填写线上 run 链接；未启用可留空
description  简短实验说明，可以写中文
```

示例：

```text
experiment	val_bpb	memory_gb	status	wandb_url	description
000-baseline-rtx5080-cuda5080	1.124884	6.0	keep		5080 基线，cuda-5080 profile
015-test-warmdown-ratio-0p62	1.118794	6.0	keep	https://wandb.ai/...	W&B online 最佳，WARMDOWN_RATIO 0.62
018-test-warmdown-ratio-0p625	1.118995	6.0	discard	https://wandb.ai/...	0.625 不如 0.62
019-test-device-batch-size-32	0.000000	0.0	crash	https://wandb.ai/...	batch 太大，OOM
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

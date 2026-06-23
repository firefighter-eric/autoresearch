# 大模型结构参数对比

截至 2026-06-23，公开资料里能直接对齐的结构参数如下。MoE 模型需要同时看 `total params` 和 `active params/token`：前者决定权重常驻显存/内存压力，后者更接近单 token 前向计算量。

| Model | 架构 | Total params | Active params/token | Layers | Hidden dim | Aspect = hidden/layers | Q heads | KV heads / KV 形式 | Head dim | Experts | Context | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |
| autoresearch current best | Dense decoder | 33.0M | 33.0M | 8 | 384 | 48.0 | 3 | 3 | 128 | - | 2,048 | `ASPECT_RATIO=48`, `DEPTH=8`, `HEAD_DIM=128`; current RTX 5080 best `val_bpb=1.101613` |
| GPT-3 175B | Dense decoder | 175B | 175B | 96 | 12,288 | 128.0 | 96 | 96 | 128 | - | 2,048 | GPT-3 paper Table 2.1 architecture family |
| Qwen3-0.6B | Dense decoder | 0.6B | 0.6B | 28 | 1,024 | 36.6 | 16 | 8, GQA | 128 | - | 32,768 | Closest official Qwen3 small text model; no official `Qwen3-0.8B` repo found |
| Qwen3-32B | Dense decoder | 32.8B | 32.8B | 64 | 5,120 | 80.0 | 64 | 8, GQA | 128 | - | 32,768 native; 131,072 YaRN | Dense Qwen3 reference point |
| Qwen3-235B-A22B | MoE decoder | 235B | 22B | 94 | 4,096 | 43.6 | 64 | 4, GQA | 128 | 128 total, 8 active | 32,768 native; 131,072 YaRN | Large Qwen3 MoE; lower active params than total params |
| Llama 4 Scout | MoE multimodal decoder | 109B | 17B | 未公开 | 未公开 | 未公开 | 未公开 | 未公开 | 未公开 | 16 | 10M | Public model card gives total/active params, experts, context; detailed text-backbone config is not public without gated access |
| Llama 4 Maverick | MoE multimodal decoder | 400B | 17B | 未公开 | 未公开 | 未公开 | 未公开 | 未公开 | 未公开 | 128 | 1M | Same 17B active budget as Scout, much larger expert pool |
| DeepSeek-V3 / DeepSeek-R1 | MoE decoder + MLA | 671B | 37B | 61 | 7,168 | 117.5 | 128 | MLA, `kv_lora_rank=512` | 128 value; QK split uses 128 no-RoPE + 64 RoPE | 256 routed + 1 shared, 8 routed active | 128K model card; config max position 163,840 | R1 uses the same main architecture family as V3; HF total package includes extra MTP weights |

## 对这个项目的直接含义

- `ASPECT_RATIO` 在本项目里对应大模型表里的 `Hidden dim` 缩放。当前从 `512` 降到 `384` 后，参数从 `50.3M` 降到 `33.0M`，5 分钟训练 token 从 `96.5M` 提到 `146.8M`。
- `HEAD_DIM` 对应表里的 `Head dim`。当前 `HEAD_DIM=128` 和主流模型一致；本轮测过 `64` 和 `256` 都更差。
- `n_head = hidden_dim / head_dim` 只适用于本项目这种简单 MHA/GQA 写法。Qwen3、DeepSeek 这类现代配置会把 `hidden_size`、Q head 数、KV 形式和 head dim 解耦。
- Dense 模型的 `active params/token ~= total params`。MoE 模型的 active params/token 明显小于 total params，但部署时仍要能放下 total params 或做专家并行/分片。

## Source Notes

- GPT-3: OpenAI, [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), Table 2.1.
- Qwen3-0.6B: [Qwen/Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) and [config.json](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json).
- Qwen3-32B: [Qwen/Qwen3-32B model card](https://huggingface.co/Qwen/Qwen3-32B) and [config.json](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json).
- Qwen3-235B-A22B: [Qwen/Qwen3-235B-A22B model card](https://huggingface.co/Qwen/Qwen3-235B-A22B) and [config.json](https://huggingface.co/Qwen/Qwen3-235B-A22B/raw/main/config.json).
- Llama 4: Meta AI, [The Llama 4 herd](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), and [Llama 4 model card](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E).
- DeepSeek-V3: [DeepSeek-V3 technical report](https://arxiv.org/html/2412.19437v1), [DeepSeek-V3 model card](https://huggingface.co/deepseek-ai/DeepSeek-V3), and [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json).

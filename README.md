# MoE agentic serving benchmark

**How many AI-agent sessions can one 8x H100 node serve on a 235B mixture-of-experts model, and what stops it?**

[![NVIDIA Dynamo 1.6.0](https://img.shields.io/badge/NVIDIA%20Dynamo-1.6.0-76B900)](https://github.com/ai-dynamo/dynamo)
[![vLLM 0.31.0](https://img.shields.io/badge/vLLM-0.31.0-1f6feb)](https://github.com/vllm-project/vllm)
[![TensorRT-LLM 1.3.0rc29](https://img.shields.io/badge/TensorRT--LLM-1.3.0rc29-76B900)](https://github.com/NVIDIA/TensorRT-LLM)
[![Model Qwen3-235B-A22B FP8](https://img.shields.io/badge/model-Qwen3--235B--A22B%20FP8%20MoE-6250d6)](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8)
[![Hardware 8x H100 SXM5](https://img.shields.io/badge/hardware-8x%20H100%20SXM5-76B900?logo=nvidia&logoColor=white)](docs/00-method.md#hardware-and-software)
[![License Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

## Two takeaways

1. **One node serves 64 concurrent agent sessions with every request in target:** first token in 0.85 s and 39 tokens/s per user (p90). That is the highest load we measured that stays in target.
2. **KV cache memory runs out before compute does.** Every turn re-sends a longer conversation, so by 256 sessions the cache is full and at most 14% of requests stay in target. An FP8 KV cache doubles capacity and throughput, the biggest lever measured.

<sub>Target: first token within 2 s and at most 50 ms per token, for at least 90% of requests.</sub>

<p align="center">
<img src="figures/anim/a05-session-growth.svg" width="100%" alt="Animation. Left: one session's prompt per turn, from 2,601 tokens at turn 0 to 8,207 at turn 7. Right: per-turn TTFT p90 and ITL p90 at 256 sessions on TP8+EP8; BF16 KV jumps at turns 4 to 5 (TTFT p90 up to 28.4 s, ITL p90 above 150 ms) while FP8 KV stays flat.">
</p>

> <sub><i>Agent prompts grow every turn, so a load that looks fine early collapses mid-session once the KV cache is full.</i></sub>

<p align="center">
<img src="figures/anim/a01-kv-cliff.svg" width="100%" alt="Animation of a 40-cell KV pool of 914,240 tokens. At 64 sessions 15 cells are used, hit 88.4%, TTFT p90 1,214 ms, ITL p90 26.7 ms. At 256 sessions the pool fills, cells are evicted and recomputed, the waiting queue reaches 113, hit falls to 52.0%, TTFT p90 rises to 17.9 s and ITL p90 to 216 ms.">
</p>

> <sub><i>Once histories no longer fit, evicted sessions are prefilled again on their next turn: the node redoes work instead of serving tokens.</i></sub>

<p align="center">
<img src="figures/anim/a06-operating-frontier.svg" width="100%" alt="Animated scatter of all 21 eight-turn agentic points: output tokens per second per GPU against tokens per second per user at p90. Solid markers meet the target, at least 90% of requests in target (all five are at 64 sessions); hollow markers do not. The frontier connects TP8+EP8 BF16 KV at 64 sessions (39 tok/s per user, 244 tok/s per GPU) and TP8+EP8 FP8 KV at 256 sessions (18 tok/s per user, 402 tok/s per GPU); the gap between them is marked as not measured. Reference lines at 20, 35 and 50 tok/s per user.">
</p>

> <sub><i>Throughput per GPU and per-user speed trade directly; only the solid points are loads you could run in production.</i></sub>

## Go deeper

| Page | What you get |
|---|---|
| [Results at a glance](docs/results.md) | Every headline number with its conditions, and how it compares with MLPerf limits and InferenceX tiers |
| [Layouts and KV capacity](docs/01-layouts-and-kv-capacity.md) | Which way to split the model across 8 GPUs |
| [The KV cliff](docs/02-load-ladder-and-kv-cliff.md) | Latency from 16 to 1,000 sessions and where it breaks |
| [vLLM vs TensorRT-LLM](docs/03-vllm-vs-tensorrt-llm.md) | Which engine to pick, phase by phase |
| [FP8 KV cache](docs/04-fp8-kv-cache.md) | What it buys and what it does not fix |
| [Prefill/decode split](docs/05-prefill-decode-disaggregation.md) | Whether to split prefill and decode on one node |
| [Sizing](docs/06-operating-frontier-and-sizing.md) | Best configuration per target, sessions to plan per node |
| [Method](docs/00-method.md) · [Limits](docs/08-limits-and-next-measurements.md) · [Glossary](docs/glossary.md) · [Reproduce](docs/reproduce.md) | How it was measured, what it does not show, and how to re-run it |

Related: [moe-disaggregation-benchmark](https://github.com/richajoy/moe-disaggregation-benchmark) (aggregated vs disaggregated serving for MoE models) · [llm-inference-performance](https://github.com/richajoy/llm-inference-performance) (which serving knob moves which metric, from first principles).

## License

[Apache-2.0](LICENSE).

# 07 How this compares

**What you get from this page:** what a "good" number is for each metric according to public references (MLPerf latency limits, InferenceX tiers and published H100 points, research conventions), where this benchmark's results sit against them, and why every comparison here is indicative only.

## The answer

At **64 sessions** this node is inside every common latency bar we found: TTFT p99 1.52 s against the 2 s conversational limit used by MLPerf for Llama2-70B and DeepSeek-R1, ITL p99 27.6 ms against per-token limits of 80 to 200 ms for conversational serving, 39 tok/s per user at p90 against the 35 tok/s per user InferenceX uses as its calculator default, and 100% of requests in target against the 90% attainment convention this benchmark uses for "meets the target". It does not reach the 50 tok/s per user InferenceX uses as its default tier. At **256 sessions with FP8 KV** it misses the 2 s TTFT bar at p99 (5.3 s) and sits at 17.7 tok/s per user. On decode economics at 64 concurrent requests, our output per GPU (244 tok/s) is close to a published H100 point for a larger MoE on a single-turn shape (235 tok/s), while our TTFT is much lower because agent prompts are mostly cached.

## Read this first: why these comparisons are indicative

- **No public benchmark covers this model.** InferenceX&trade; (formerly InferenceMAX), SemiAnalysis's open inference benchmark, has no data for Qwen3-235B-A22B ([hardware and model lists](https://inferencex.semianalysis.com/api/v1/views/options)); every InferenceX number here is for a different model.
- **Our agentic load is not AgentX.** It is a synthetic multi-turn shape (2,600 to 8,200-token prompts, 200-token outputs, a 2,000-token prefix shared by all sessions). InferenceX AgentX replays recorded coding-agent sessions whose median request has 142,016 input tokens ([AgentX methodology](https://inferencex.semianalysis.com/agentx/methodology)), about 17x our longest prompt. Sessions per node do not carry over between the two.
- **MLPerf: limit values only.** MLCommons does not allow MLPerf results to be compared with non-MLPerf results ([messaging guidelines](https://github.com/mlcommons/policies/blob/master/MLPerf_Results_Messaging_Guidelines.adoc)). We cite MLPerf's published latency **limit values** as a common bar, nothing else. MLPerf Server load is open-loop Poisson for 600 s; ours is closed-loop concurrency.
- **Same statistic or no comparison.** InferenceX computes P90 interactivity as 1000 / P90 ITL, the slow tail ([glossary](https://inferencex.semianalysis.com/glossary/p90-interactivity)); our "tok/s per user (p90)" uses the same formula. Its fixed-sequence charts use the median, so where we compare with those we show our median. MLPerf limits are at p99, so we show our p99 next to them.
- **Output vs total tokens.** InferenceX headline throughput is total (input + output) tokens per GPU, which on agentic traces is dominated by cached input ([glossary](https://inferencex.semianalysis.com/glossary/throughput)). We headline output tokens per GPU and give total separately.

## Latency bars

**MLPerf Inference datacenter LLM limits** (p99; [rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)). TPOT there is the average interval between a request's tokens, the same per-request quantity as our ITL.

| Model in the MLPerf rules | Conversational / Server: TTFT, TPOT | Interactive: TTFT, TPOT |
|---|---|---|
| Llama2-70B | 2,000 ms, 200 ms | 450 ms, 40 ms |
| Llama3.1-8B | 2,000 ms, 100 ms | 500 ms, 30 ms |
| Mixtral-8x7B (MoE) | 2,000 ms, 200 ms | none listed |
| Llama3.1-405B (long context) | 6,000 ms, 175 ms | 4,500 ms, 80 ms |
| DeepSeek-R1 (MoE) | 2,000 ms, 80 ms | 1,500 ms, 15 ms |
| GPT-OSS-120B (MoE) | 3,000 ms, 80 ms | 2,000 ms, 20 ms |

Our p99 values against those bars:

| Our point | TTFT p99 | ITL p99 | Under the 2 s conversational TTFT bar? | Under the 80 ms MoE server TPOT bar? |
|---|---|---|---|---|
| TP8+EP8, BF16 KV, 64 sessions | 1,521 ms | 27.6 ms | yes | yes (also under the 40 ms Llama2-70B interactive bar) |
| P/D, FP8 KV, 64 sessions | 1,983 ms | 27.1 ms | yes, barely | yes |
| TP8+EP8, FP8 KV, 256 sessions | 5,309 ms | 61.1 ms | no (under the 6 s long-context bar) | yes |
| TP8+EP8, BF16 KV, 256 sessions | 28,448 ms | 166.5 ms | no | no |
| TP8+EP8, FP8 KV, 1,000 sessions | 83,309 ms (lower bound) | 280.0 ms | no | no |

**Human perception** ([Nielsen Norman Group, response-time limits](https://www.nngroup.com/articles/response-times-3-important-limits/)): about 1 s keeps a user's flow of thought uninterrupted, 10 s is the limit of attention. Our 64-session TTFT p90 (851 ms) is inside 1 s; at 256 sessions with FP8 KV (4.1 s) it is not; at 1,000 sessions (72.5 s or more) it is past the attention limit.

## Per-user speed tiers

| Reference | Value | Type | Ours (p90) |
|---|---|---|---|
| Our ITL bound, 50 ms | 20 tok/s per user | convention, arithmetic | 64 sessions: 39.0; FP8 KV at 256: 17.7 |
| InferenceX calculator default ([v2 article](https://newsletter.semianalysis.com/p/inferencex-v2-nvidia-blackwell-vs): "the middle of the pack provider serves at an interactivity of around 35 tok/s/user") | 35 tok/s per user | benchmark default | above it at 64 sessions |
| InferenceX overview tiers ([options API](https://inferencex.semianalysis.com/api/v1/views/options)) | 30, 50 (default), 75, 100, 150, 200 | benchmark tiers | between 30 and 50; no 8-turn point reaches 50 |
| InferenceX first-token-limits page agentic floor ([page](https://inferencex.semianalysis.com/first-token)) | 150 tok/s per user | UI default, not an SLO | far below; would need speculative decoding or much lower concurrency, neither measured |

## Published H100 points for other MoE models (InferenceX)

**Fixed sequence 8k/1k**, Qwen3.5-397B-A17B FP8 on 8x H100 with SGLang, TP8+EP8 at 16 to 256 concurrent, median statistics ([data](https://inferencex.semianalysis.com/api/v1/views/inference?model=Qwen-3.5-397B-A17B&sequence=8k-1k&metric=y_tpPerGpu&gpus=h100&optimal=false&best=false)). Its prompts are 6,553 to 8,192 tokens with outputs of 819 to 1,024; ours are 2,600 to 8,200 tokens with about 88% served from cache and 200-token outputs.

| Concurrent | InferenceX: tok/s per user (median) | InferenceX: output tok/s/GPU | InferenceX: median TTFT | Ours, TP8+EP8 BF16 KV: tok/s per user (median) | Ours: output tok/s/GPU | Ours: median TTFT |
|---|---|---|---|---|---|---|
| 64 | 39.8 | 235 | 7.90 s | 44.9 | 244 | 0.56 s |
| 256 | 10.7 | 285 | 13.79 s | 17.5 (FP8 KV: 22.2) | 203 (FP8 KV: 402) | 3.34 s (FP8 KV: 2.06 s) |

What it suggests: at 64 concurrent, per-GPU decode output and per-user speed are in the same range for a 22B-active and a 17B-active MoE on the same GPUs; the TTFT gap is the agentic prefix cache, not a faster engine. Different model, engine and workload: indicative only.

**AgentX on H100**, Qwen3.5-397B-A17B FP8 with SGLang, TP8+EP8, speculative decoding (MTP), P90 statistics ([data](https://inferencex.semianalysis.com/api/v1/views/inference?model=Qwen-3.5-397B-A17B&sequence=agentic&metric=y_tpPerGpu&gpus=h100&optimal=false&best=false)): P90 TTFT is about 1.4 to 1.5 s at 1 to 4 sessions and 4.45 s at 8 sessions without KV offload; with CPU KV offload, 12 sessions reach 88.7 tok/s per user at P90 with P90 TTFT 12.15 s; without offload 12 to 16 sessions collapse (P90 TTFT 67 to 85 s). Its measured prefix-cache hit at the 12-session frontier point is 92.5%. Those sessions carry contexts about 17x longer than ours and the run used speculative decoding, so the session counts are not comparable; what carries over is the shape of the result: on H100, agent capacity is bounded by KV capacity, and host-memory offload is what extends it.

## SLO attainment and prefix reuse

| Reference | Bar | Ours |
|---|---|---|
| Goodput convention ([DistServe](https://arxiv.org/abs/2401.09670)), the rule used here | at least 90% of requests within both latency targets | 100% at 64 sessions on TP8+EP8; 18% at 256 with FP8 KV; 0% at 1,000. Five of the nine 8-turn points at 64 sessions qualify; none above 64 does |
| MLPerf constraint level | 99th percentile | the same five 64-session points (lowest 99.6%); no agentic point falls between 90% and 99%, so the stricter bar changes nothing here |
| Prefix-cache hit vs its theoretical ceiling ([AIPerf metrics](https://github.com/ai-dynamo/aiperf/blob/main/docs/metrics-reference.md)) | at the ceiling the workload allows | 88.2% vs 88.9% at 64 sessions; 89.0% vs 89.7% at 256 with FP8 KV; 37.6 to 57.0% once the pool is full |

## What to do with it

- **Use the 64-session configuration if you need latency comparable to common conversational bars**; it clears them with margin at p99.
- **Do not quote 402 tok/s per GPU as an interactive number.** It sits at 17.7 tok/s per user and 18% in target, below the lowest InferenceX tier.
- **To reach 50+ tok/s per user** on this model and node you would need lower concurrency, speculative decoding, or both; neither was measured for the 8-turn agentic shape here.
- **For a like-for-like comparison with AgentX**, run AIPerf's AgentX scenario ([tutorial](https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/inference-x-agent-x-mvp-benchmark)) and label the result unofficial, as InferenceX requires for runs outside its own repository ([README](https://github.com/SemiAnalysisAI/InferenceX)).

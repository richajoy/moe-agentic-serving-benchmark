# Glossary

**What you get from this page:** a plain-language definition of every term and abbreviation used in this benchmark, with what one of its numbers means here.

## The answer

Four terms carry most of this benchmark. **TTFT** is how long a request waits for its first token; **ITL** is the average gap between its following tokens; a request is **in target (SLO-ok)** when TTFT &le; 2 s and its mean ITL &le; 50 ms; a point **meets the target** when at least 90% of its requests are in target; **goodput** is how many in-target requests per second each GPU serves. Layouts are written as **TP8+EP8** (attention split over 8 GPUs, experts spread over 8 GPUs), and **P/D** means separate prefill and decode workers. Headline percentiles are **p90**.

## Notation

| Term | Meaning |
|---|---|
| `c64`, `c1000`, "64 sessions" | Concurrency: the load generator keeps 64 or 1,000 sessions open at once, closed loop (a session sends its next turn only after the previous one finishes and its tool gap passes) |
| `AG-c256-t8` | A point name: agentic workload (AG), 256 concurrent sessions, 8 &plusmn; 2 turns per session |
| `S1-c64`, `S2-c16`, `S3-c16` | Single-turn fixed-shape points at the given concurrency |
| `t0`, `t5` | Turn index within a session, counted from 0; `t5` is the sixth turn |
| p50 / p90 / p99 | Median, 90th and 99th percentile over the requests of one point. **p90 is the headline percentile** in this benchmark; tables show p90 first, then (p50 / p99) |
| A &rarr; B | The value before and after one change |
| k, M | Thousand, million (tokens unless stated) |
| GiB vs GB | 2^30 bytes (engine logs) vs 10^9 bytes (transfer sizes) |

## Model and hardware

| Term | Meaning |
|---|---|
| **Qwen3-235B-A22B-Instruct-2507** | The model: 235 billion parameters in total, about 22 billion used per token, served from its FP8 checkpoint |
| **MoE** (mixture of experts) | Each MoE layer holds many small feed-forward networks (experts) and a router that sends each token to a few of them. Here 128 experts per layer, 8 used per token |
| **FP8** | 8-bit floating point. The weights use FP8 with one scale per 128 x 128 block; the KV cache can optionally be FP8 too |
| **BF16** | 16-bit floating point; the default KV cache format here |
| **H100 SXM5** | NVIDIA H100 80 GB in the SXM form factor, connected to the other 7 GPUs by NVLink |
| **NVLink** | The direct GPU-to-GPU interconnect; every GPU reaches every other at full speed on this node |

## Parallelism and layouts

| Term | Meaning |
|---|---|
| **TP** (tensor parallel), **TP8** | Every layer is split across 8 GPUs that compute each token together. With 4 KV heads on 8 GPUs, each GPU stores one KV head per token |
| **EP** (expert parallel), **EP8** | Whole experts are spread across GPUs (128 / 8 = 16 per GPU) and tokens are sent to the GPU holding their expert |
| **TP8+EP8** | Attention split 8 ways (TP8), experts spread 8 ways (EP8). vLLM: `--tensor-parallel-size 8 --enable-expert-parallel`. The recommended layout here |
| **DP** (data parallel), **DP8+EP8** | 8 independent attention replicas (ranks), each serving its own requests with its own KV cache, sharing experts spread EP8 |
| **Aggregated** | One engine does both prefill and decode for every request |
| **Prefill** | Processing a request's prompt to build its KV cache and produce the first token; compute-heavy |
| **Decode** | Generating output tokens one step at a time, each step reading the KV cache; memory-bandwidth-heavy at small batch |
| **P/D** (prefill/decode disaggregation) | Separate prefill and decode workers; the prefill worker's KV is copied to the decode worker |
| **1P:1D** | One prefill worker and one decode worker; here each is TP4+EP4 on 4 GPUs |
| **Like-for-like** | A comparison where exactly one setting differs and both points used the same completion rule |

## Workloads and completion rules

| Term | Meaning |
|---|---|
| **AG** (agentic) | Multi-turn sessions: a 2,000-token system prompt shared by every session, +600 new input and 200 output tokens per turn, a 1,000 &plusmn; 250 ms tool gap, the whole history re-sent each turn. Prompts grow from 2,600 tokens (turn 0) to 8,200 (turn 7) |
| **Session, turn** | A session is one simulated agent; a turn is one request in it |
| **Tool gap** | Simulated time an agent spends running a tool between a response and its next request |
| **S1, S2, S3** | Single-turn shapes: 512 in / 4,096 out (decode-bound); 16,384 in / 256 out (prefill-bound); 16,384 in / 4,096 out |
| **ISL, OSL** | Input and output sequence length in tokens |
| **count-capped** | A point that stops after a fixed number of requests; the longest sessions' last turns are missing |
| **time-capped** | A point that stops at a time limit plus a grace period. **Complete** if every planned turn ran; otherwise it **stopped early** and its latencies are **lower bounds** |
| **Lower bound** | A latency the full workload would exceed, because the run ended while later, slower turns were still pending |
| **Effective concurrency** | Time-averaged requests in flight; lower than `c` because sessions idle during the tool gap |

## Latency, throughput and the target

| Term | What one number means here |
|---|---|
| **TTFT** (time to first token) | Wait from sending a request to its first streamed token: queueing + prefill (+ KV transfer for P/D). "TTFT p90 851 ms" = 90% of requests saw their first token within 851 ms |
| **ITL** (inter-token latency) | A request's mean gap between output tokens: (request latency - TTFT) / (output tokens - 1). Percentiles are across requests. Also called TPOT (time per output token). 25 ms = 40 tokens per second |
| **tok/s per user (p90)**, interactivity | 1000 / ITL p90: the streaming speed of the slowest 10% of requests. "39 tok/s per user" = 90% of requests streamed at least that fast. Excludes the wait for the first token |
| **E2E** | End-to-end request latency: TTFT plus all output tokens |
| **req/s** | Completed requests (turns) per second |
| **output tok/s per GPU** | Output tokens per second divided by 8 GPUs (P/D included) |
| **total tok/s per GPU** | Input plus output tokens per second divided by 8; on agentic load mostly re-sent, cached input |
| **SLO** (service-level objective) | The latency target: TTFT &le; 2 s **and** the request's mean ITL &le; 50 ms |
| **SLO-ok, "in target"** | A request that met the SLO; as a number, the share of measured requests that met it |
| **Meets the target** (a point) | At least 90% of the point's requests are SLO-ok (the [DistServe](https://arxiv.org/abs/2401.09670) goodput convention; MLPerf constrains the 99th percentile instead). Solid markers in the frontier figure |
| **Goodput** | Requests per second per GPU that met the SLO: req/s x SLO-ok share / 8. Compared across points that meet the target |
| **Frontier** (Pareto frontier) | The points no other point beats on both output per GPU and per-user speed |
| **Knee** | The load past which latency grows much faster than throughput (between 64 and 256 sessions here) |

## KV cache

| Term | Meaning |
|---|---|
| **KV cache** | The attention keys and values of every token already processed, kept on the GPU so they are not recomputed. 188 KiB per token for this model in BF16 |
| **KV pool, KV capacity** | How many tokens of KV fit in GPU memory after weights, from the engine's startup log (for example 906,304 on TP8+EP8 with BF16 KV) |
| **KV peak** | The highest share of the pool in use during a point; 1.00 means full |
| **Block** | KV is managed in fixed blocks of 64 tokens here |
| **Prefix caching, prefix hit** | Reusing the KV of an identical prompt prefix. Hit = share of prompt tokens served from cache. TensorRT-LLM counts blocks instead of tokens |
| **Ideal hit** | The hit if nothing is ever evicted and only each turn's 600 new tokens miss: 88.9% for 8-turn sessions |
| **Eviction** | Freeing cached blocks (here, idle sessions' history) to make room |
| **Preemption** | The engine drops a running request's KV and recomputes it later |
| **KV cliff** | The load at which sessions outgrow the pool, so hits, queueing and latency break together |
| **FP8 KV** | Storing the KV cache in 8-bit floats (`--kv-cache-dtype fp8`): half the bytes, twice the tokens. **Uncalibrated** here: the KV scale was 1.0, so output quality is not verified |
| **Session demand** | KV a session holds; about 8,400 tokens at the end of an 8-turn session |

## Engines, kernels and transfer

| Term | Meaning |
|---|---|
| **NVIDIA Dynamo** | NVIDIA's inference serving framework: an OpenAI-compatible frontend, routers, engine workers and KV transfer (version 1.6.0) |
| **Frontend, round-robin, KV router** | The Dynamo HTTP entry point; `round-robin` cycles across workers, `kv` sends each request to the worker or rank holding the most of its cached prefix |
| **vLLM, TensorRT-LLM** | The two inference engines compared (0.31.0 and 1.3.0rc29) |
| **Chunked prefill, step budget** | Long prompts are split into chunks that share engine steps with running decodes; vLLM's budget is 8,192 tokens per step by default |
| **Engine step** | One scheduler iteration: a batch of decode tokens plus any prefill chunk |
| **Triton fused MoE, default MoE config** | vLLM's Triton MoE kernel uses per-shape tuned config files; without one it logs `Using default MoE config` |
| **FLASHINFER_CUTLASS** | The MoE kernel vLLM picks under expert parallelism here |
| **Block refinement** | On pure TP8 vLLM re-cuts the 128-wide FP8 scale blocks to [64, 64] because each GPU's 192-wide expert slice is not a multiple of 128 |
| **NIXL** | NVIDIA's library that moves KV between the prefill and decode workers, here over NVLink |
| **Transfer, descriptor, post time** | One request's KV for one decode GPU; a descriptor is one contiguous memory region in it; post time is the CPU time to submit the descriptors |
| **AIPerf** | NVIDIA's open-source benchmark client (0.13.0) that generated every load and recorded every request |
| **InferenceX, AgentX** | SemiAnalysis's open inference benchmark and its coding-agent trace workload; used here only as an indicative reference ([docs/07](07-how-this-compares.md)) |

## What to do with it

When you quote a number from this benchmark, carry its percentile (p90 unless stated), its point name (layout, KV dtype, sessions) and its completion rule; together they say what the number means and what it can be compared with.

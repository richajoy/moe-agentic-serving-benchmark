# 00 Method

**What you get from this page:** exactly what was run (hardware, software, model, workload, target), how each point ends, how every column is computed, and the caveats you need before you reuse a number.

## The answer

Every number in this benchmark is one AIPerf 0.13 run against an NVIDIA Dynamo 1.6.0 frontend on one 8x H100 SXM5 node, serving Qwen3-235B-A22B-Instruct-2507 with FP8 weights. The main workload is multi-turn agentic sessions whose prompt grows from 2,600 to 8,200 tokens over 8 turns. A request counts as **in target (SLO-ok)** when its TTFT is at most 2 s **and** its mean ITL is at most 50 ms; a point **meets the target** when at least 90% of its requests are in target. Headline percentiles are **p90**. Three method limits apply to every page: single runs, mixed completion rules, and no prefix-cache reset between points on the same server.

- [Hardware and software](#hardware-and-software)
- [Model](#model)
- [Engine settings](#engine-settings)
- [Workloads](#workloads)
- [The target (SLO) and why these bounds](#the-target-slo-and-why-these-bounds)
- [Completion rules](#completion-rules)
- [How each column is computed](#how-each-column-is-computed)
- [KV cache arithmetic](#kv-cache-arithmetic)
- [Caveats that apply to every page](#caveats-that-apply-to-every-page)
- [What to do with it](#what-to-do-with-it)

## Hardware and software

| Item | Value |
|---|---|
| GPUs | 8x NVIDIA H100 80GB HBM3 (SXM5), driver 580.105.08 |
| GPU interconnect | NVLink, full mesh between all 8 GPUs |
| CPU / memory | 2 NUMA nodes (GPUs 0-3 and 4-7), 208 vCPUs, 1.77 TB RAM |
| Network | one NIC with no GPUDirect path; irrelevant on one node |
| Containers | Docker with the NVIDIA container toolkit; no Kubernetes |
| Serving | NVIDIA Dynamo 1.6.0 (`vllm-runtime:1.6.0`, `tensorrtllm-runtime:1.6.0` images) |
| Engines | vLLM 0.31.0; TensorRT-LLM 1.3.0rc29 (PyTorch backend) |
| Libraries | torch 2.13.0+cu130, CUDA 13.0, Triton 3.7.1, NIXL 1.4.1 |
| Load generator | AIPerf 0.13.0 on the same node, against `localhost:8000` |

**Interconnect baseline** (NCCL, 8 ranks, `scripts/analysis/nccl_bandwidth.py`): all-reduce bus bandwidth 46 GB/s at 1 MB, 247 at 16 MB, 433 at 256 MB, 467 at 1 GB; all-to-all algorithm bandwidth 15, 198, 314 and 329 GB/s at the same sizes. Bus and algorithm bandwidth are different metrics and are not directly comparable.

## Model

| Item | Value |
|---|---|
| Model | Qwen3-235B-A22B-Instruct-2507, FP8 checkpoint, served as `qwen235` |
| Size | 235B total parameters, about 22B active per token (mixture of experts) |
| Layers / attention | 94 decoder layers, 4 KV heads of dimension 128 (grouped-query attention) |
| Experts | 128 experts per MoE layer, 8 active per token, expert intermediate size 1536 |
| Weight quantization | FP8 with one scale per 128 x 128 block |

## Engine settings

All vLLM runs share one base command; each run changes one or two flags. Exact commands: [data/configs/](../data/configs/).

```
python -m dynamo.vllm --model <Qwen3-235B-A22B-Instruct-2507-FP8> --served-model-name qwen235 \
  --max-model-len 40960 --gpu-memory-utilization 0.92 --block-size 64 --max-num-seqs 1024 \
  --enable-prefix-caching --tensor-parallel-size 8
python -m dynamo.frontend --http-port 8000 --router-mode round-robin --kv-cache-block-size 64
```

| Run (data directory) | Change from the base | KV pool (tokens, from the startup log) |
|---|---|---|
| `vllm-tp8`, `vllm-tp8-matrix` | none (TP8); two launches of the same command | 914,240 / 914,688 |
| `vllm-tp8ep8` | `--enable-expert-parallel` | 906,304 |
| `vllm-tp8ep8-fp8kv` | `--enable-expert-parallel --kv-cache-dtype fp8` | 1,812,224 |
| `vllm-dp8ep8` | `--data-parallel-size 8 --enable-expert-parallel`, `--max-num-seqs 256` per rank | 8 x 93,376 = 747,008 |
| `vllm-dp8ep8-kvrouter` | as above, plus KV events and frontend `--router-mode kv` | 8 x 93,504 = 748,032 |
| `trtllm-tp8ep8` | TensorRT-LLM worker, [deploy/trtllm-tp8ep8.yaml](../deploy/trtllm-tp8ep8.yaml) | 860,608 |
| `vllm-pd-1p1d`, `vllm-pd-1p1d-fp8kv` | decode TP4+EP4 on GPUs 0-3, prefill TP4+EP4 on GPUs 4-7, NIXL KV transfer; BF16 or FP8 KV | decode 294,016 / 587,648 |
| `vllm-tp8-moe-default`, `vllm-tp8-moe-tuned` | TP8 relaunched without and with a tuned fused-MoE config | 914,688 |

Defaults recorded from the startup logs: FlashAttention 3 attention on every vLLM run; FP8 MoE kernel TRITON on pure TP8 and FLASHINFER_CUTLASS under expert parallel; chunked prefill with 8,192 tokens per step; TensorRT-LLM capacity scheduler `GUARANTEED_NO_EVICT`.

## Workloads

| ID | Definition | Purpose |
|---|---|---|
| **AG** (agentic, `t8`) | 2,000-token shared system prompt; each turn adds 600 new input tokens and produces 200 output tokens; 1,000 &plusmn; 250 ms tool gap between turns; 8 &plusmn; 2 turns per session; one session per simulated user; sessions = max(concurrency, 64) | the main workload: sessions whose context grows |
| AG `t4`, `t20` | as AG with 4 &plusmn; 1 or 20 &plusmn; 5 turns | smoke point; session-length sensitivity |
| S1 | 512 input / 4,096 output tokens, single turn | decode-bound baseline |
| S2 | 16,384 input / 256 output | prefill-bound baseline |
| S3 | 16,384 input / 4,096 output | long both |

**Each agentic turn re-sends the whole conversation**, so the prompt grows by 800 tokens per turn (600 new input plus the previous 200 output). Measured median prompt length by turn (tokenizer count, identical across runs):

| Turn (0-based) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| Prompt tokens | 2,601 | 3,402 | 4,203 | 5,004 | 5,804 | 6,605 | 7,406 | 8,207 | 9,008 | 9,809 |

- Output length is fixed at 200 tokens with `ignore_eos` and `min_tokens`, so output tok/s = req/s x 200 on every agentic point.
- The realized sessions average 8.5 to 8.9 turns, not exactly 8 (normal distribution, stddev 2).
- One request is one turn. Concurrency `c` is the number of sessions the load generator keeps open (closed loop). Because sessions idle during the tool gap, the time-averaged number of requests in flight is lower: about 49 to 51 at c64, 179 to 221 at c256, 819 and 907 at c1000.
- Seed 42 on every point. No warm-up phase was configured, so every request of a point counts in its percentiles, SLO-ok and goodput. The fixed shapes on `vllm-tp8-matrix` S1 reused prompts from a small synthetic pool (S1 at 64 concurrent hit 98.5% of the prefix cache), so S1 TTFT on that run is not valid; ITL and throughput are.

## The target (SLO) and why these bounds

A request is **SLO-ok** when TTFT &le; 2,000 ms **and** its mean ITL &le; 50 ms. Both are judged per request, so one point can have a good median and still score 0%.

A point (one load on one configuration) **meets the target** when **at least 90% of its requests are SLO-ok**. That is the goodput convention of [DistServe](https://arxiv.org/abs/2401.09670) and the rule used by the sibling [moe-disaggregation-benchmark](https://github.com/richajoy/moe-disaggregation-benchmark). Because a request must meet both bounds at once, meeting the target in effect puts TTFT p90 and ITL p90 inside the bounds; the reverse does not hold. MLPerf's constraints sit at the 99th percentile instead; in this data set no agentic point lands between 90% and 99% (the lowest passing point is 99.6%, the highest failing one 87.5%), so the stricter bar would not change any agentic result. `python3 scripts/analysis/check_data.py` prints the yes/no for every point.

| Bound | Where it comes from | What it means for a user |
|---|---|---|
| TTFT &le; 2 s | The MLPerf Inference datacenter limit for conversational LLMs (Llama2-70B, DeepSeek-R1) is a 2 s TTFT at p99 ([rules](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc)); we use the value as a common bar, not as an MLPerf result | an agent step waits at most 2 s before output starts |
| ITL &le; 50 ms | Arithmetic: 50 ms per token = 20 tok/s per user. No standard body uses 50 ms; MLPerf's per-token limits range from 15 ms (interactive) to 200 ms (conversational) | the stream never falls below 20 tokens per second |

No public body sets an agentic SLO ([InferenceX glossary](https://inferencex.semianalysis.com/glossary/service-level-objective): "InferenceX does not impose one industry SLO"). [docs/07](07-how-this-compares.md) puts every bar next to our numbers.

## Completion rules

| Rule | How the point ends | Points |
|---|---|---|
| **count-capped** | stops after sessions x mean turns requests; the longest sessions' last turns are missing | every 64-session 8-turn point (512 of 572 planned turns, 89.5%), 16 sessions (256 of 301), and the 256-session points on `vllm-tp8`, `vllm-dp8ep8*` (2,048 of 2,208) |
| **time-capped, complete** | 300 s + 120 s grace, and every planned turn ran | 256 sessions on `vllm-tp8ep8`, `trtllm-tp8ep8`, `vllm-tp8ep8-fp8kv` (2,208 of 2,208); 64 sessions with 20 turns (1,378 of 1,378) |
| **time-capped, stopped early** | the cap hit while sessions were still running; latencies are **lower bounds** | 512 sessions (3,179 of 4,367), both 1,000-session points (4,122 and 5,246 of 8,550), P/D at 256 (2,152 of 2,208), the two MoE-config runs at 256 (240 s cap: 1,905 and 2,003 of 2,208) |
| **fixed request count** | max(c, 100) single-turn requests | S1, S2, S3 |

What this means for comparisons: all 64-session points are comparable with each other (same rule). BF16 vs FP8 KV at 256 sessions on TP8+EP8 is like-for-like (both complete). TP8 vs TP8+EP8 at 256 sessions is not (count-capped vs complete: the complete run includes cheap late turns that lower its percentiles). Each page names the rule.

## How each column is computed

| Column | Computation |
|---|---|
| req/s | completed requests (turns) per second over the profiling window (AIPerf `request_throughput`) |
| output tok/s per GPU | AIPerf output token throughput / 8. Every run used 8 GPUs, including P/D (4 prefill + 4 decode) |
| total tok/s per GPU | input + output tokens per second / 8. Input includes the re-sent, mostly cached history, so it is large on agentic points; quoted only when comparing with benchmarks that report total tokens |
| TTFT p50 / p90 / p99 | time from sending a request to its first streamed token, percentiles over the requests of one point |
| ITL p50 / p90 / p99 | a request's mean gap between output tokens, (request latency - TTFT) / (output tokens - 1) as AIPerf computes it; percentiles across requests |
| **tok/s per user (p90)** | **1000 / ITL p90**: the streaming speed of the slowest 10% of requests. This is how InferenceX computes P90 interactivity ([glossary](https://inferencex.semianalysis.com/glossary/p90-interactivity)). It is **not** AIPerf's per-user throughput p90, which is the fast tail |
| SLO-ok % | share of measured requests with TTFT &le; 2,000 ms and mean ITL &le; 50 ms; the point meets the target at &ge; 90% |
| goodput per GPU | req/s x SLO-ok share / 8: requests per second per GPU that met the target |
| prefix-cache hit % | vLLM: cached prompt tokens / queried prompt tokens. TensorRT-LLM: reused / (reused + missed) 64-token blocks. P/D: both pools summed |
| KV peak | the highest share of KV blocks in use during the point (1.00 = full); cached-but-free blocks do not count |
| preemptions, waiting p90 | vLLM `num_preemptions` total and the p90 of the `num_requests_waiting` gauge; TensorRT-LLM does not export them |

**Ideal prefix-cache hit.** If nothing is ever evicted, only each turn's 600 new tokens miss the cache: hit = 1 - 600n / (sum of the n turn prompts). That is 84.2% for 4 turns, **88.9% for 8 turns**, 90.3% for 10 and 94.1% for 20. For the sessions actually generated at 256 concurrent (8.6 turns on average) the ceiling is 89.7%.

Every column is reproducible from [data/](../data/README.md): `python3 scripts/analysis/check_data.py` recomputes SLO-ok and goodput from the per-request records and checks every percentile against the AIPerf summaries.

## KV cache arithmetic

- BF16 KV costs 94 layers x 4 KV heads x 128 x 2 (K and V) x 2 bytes = 192,512 bytes (188 KiB) per token for the whole model; FP8 KV half that.
- TP8 splits 4 KV heads over 8 GPUs, so each GPU stores one head per token: about 47 KiB per token per GPU. 41 GiB of free memory per GPU / 47 KiB = about 914k tokens, matching the log.
- DP8+EP8 stores all 4 heads on each rank for its own requests (188 KiB per token per rank) and replicates attention weights, so it holds less in total (747k).
- **Session demand:** an 8-turn session ends near 8,400 tokens (8,207 prompt + 200 output). 256 sessions at full length is about 2.15M tokens (1.64M if the 2,000-token shared prompt is stored once); 1,000 sessions is 6.4M to 8.4M. These end-of-session figures overstate what is resident at any one moment, because sessions are at different turns.

## Caveats that apply to every page

1. **Single runs** (n = 1). Treat differences under about 10% as noise. The one repeat available: a relaunch of the TP8 command reproduced 64 sessions at 9.26 vs 9.29 req/s and TTFT p90 1,230 vs 1,214 ms, and S2 at 16 concurrent at 1.00 req/s both times. Tails move more than medians between runs.
2. **No prefix-cache reset** between points on the same server, and seed 42 is reused, so an early session of a later point can hit blocks left by an earlier point.
3. **Mixed completion rules** (above); each comparison says whether it is like-for-like.
4. **Lower bounds** at 512 and 1,000 sessions and for P/D at 256.
5. **FP8 KV is uncalibrated.** The checkpoint has no KV scales, vLLM uses 1.0 and warns about possible accuracy loss. Output quality with FP8 KV was not checked.
6. **TensorRT-LLM** counts prefix hits in blocks and exports no preemption or queue gauges; its KV pool is 5% smaller than vLLM's at the same layout.
7. **Gaps**: no point between 64 and 256 sessions (96 to 192) was run on any configuration; FP8 KV was not run at 64.

## What to do with it

- Quote a number together with its run, point and completion rule (for example "TP8+EP8, BF16 KV, 64 sessions, count-capped"). The `source` field in [data/points.json](../data/points.json) gives the path.
- If you rerun this, use one completion rule for every point, reset or restart the server between points, and repeat each point at least three times; that changes error bars more than rankings.
- If your workload has longer sessions or a different shared prefix, recompute the session-demand arithmetic above before reusing the session counts on the other pages.

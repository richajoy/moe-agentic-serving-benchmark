# 06 Operating frontier and sizing

**What you get from this page:** the measured trade-off between throughput per GPU and per-user speed for agentic serving on this node, the best configuration for each kind of target, a one-screen map of which change moves which metric, and how many sessions (and nodes) to plan for.

## The answer

Only **two measured points are worth running**, and they sit at opposite ends:

- **For an interactive target** (TTFT &le; 2 s and mean ITL &le; 50 ms for at least 90% of requests): vLLM TP8+EP8, BF16 KV, **64 sessions per node**: TTFT p90 851 ms, ITL p90 25.6 ms, 39.0 tok/s per user at p90, 244 output tok/s per GPU, 100% in target.
- **For throughput** (batch agents that tolerate multi-second first tokens): vLLM TP8+EP8, FP8 KV, **256 sessions per node**: 402 output tok/s per GPU (1.65x more), but TTFT p90 4.1 s, ITL p90 56.4 ms (17.7 tok/s per user) and 18% in target, with FP8 KV quality unverified.

Every other measured point is beaten on both axes by one of these. The best point for most real targets, the highest throughput that still keeps at least 90% of requests in target, is almost certainly **between 64 and 256 sessions, which was not measured**.

**What a good number is:** InferenceX&trade; treats 35 tok/s per user as a reasonable speed (its calculator default) and uses 50 as its default tier; our ITL bound corresponds to 20. These tiers come from a different benchmark, and our synthetic multi-turn load is not its AgentX workload ([docs/07](07-how-this-compares.md)). The interactive point clears 35, not 50; the throughput point clears neither.

## The frontier

<p align="center">
<img src="../figures/anim/a06-operating-frontier.svg" width="100%" alt="Animated scatter of all 21 eight-turn agentic points: output tokens per second per GPU against tokens per second per user at p90. Solid markers meet the target, at least 90% of requests in target (all five are at 64 sessions); hollow markers do not. The frontier connects TP8+EP8 BF16 KV at 64 sessions (39 tok/s per user, 244 tok/s per GPU) and TP8+EP8 FP8 KV at 256 sessions (18 tok/s per user, 402 tok/s per GPU); the gap between them is marked as not measured. Reference lines at 20, 35 and 50 tok/s per user.">
</p>

> <sub><i>Throughput per GPU and per-user speed trade directly, and the measured curve has a hole exactly where most deployments would want to operate.</i></sub>

| Frontier point | tok/s per user (p90) | output tok/s per GPU | TTFT p90 (p50 / p99) | ITL p90 (p50 / p99) | in target | goodput per GPU |
|---|---|---|---|---|---|---|
| TP8+EP8, BF16 KV, 64 sessions | **39.0** | 243.6 | **851 ms** (565 / 1,521) | **25.6 ms** (22.3 / 27.6) | 100% | **1.218** |
| TP8+EP8, FP8 KV, 256 sessions | 17.7 | **401.9** | **4.10 s** (2.06 / 5.31) | **56.4 ms** (45.1 / 61.1) | 18.4% | 0.369 |

The two points differ in **both** concurrency and KV dtype. The frontier is computed over all 21 eight-turn agentic points with `python3 scripts/analysis/check_data.py --frontier` (non-dominated on output tok/s per GPU and 1000 / ITL p90).

**Best output per GPU at a per-user speed floor** (p90 basis):

| Floor (tok/s per user at p90) | ITL p90 must be &le; | Best measured point | Also meets the target (&ge; 90% of requests in target)? |
|---|---|---|---|
| 20 (our ITL bound) | 50 ms | TP8+EP8 BF16 KV, 64 sessions: 243.6 tok/s/GPU | yes |
| 30 | 33.3 ms | same | yes |
| 35 (InferenceX calculator default) | 28.6 ms | same | yes |
| 40 | 25.0 ms | **none measured**; the fastest 8-turn point is 39.0 | n/a |
| 50 (InferenceX default tier) | 20.0 ms | **none measured** | n/a |

Note that at the p90 basis the FP8 KV point (17.7 tok/s per user) misses even the 20 tok/s floor; it is the throughput end of the curve, not an interactive operating point.

## Best configuration per target

| Target | Use | Measured at | Risk to check first |
|---|---|---|---|
| Interactive: TTFT &le; 2 s and ITL &le; 50 ms for &ge; 90% of requests | vLLM TP8+EP8, BF16 KV | 64 sessions: 100% in target, goodput 1.22 req/s per GPU | the limit above 64 is unknown; FP8 KV at 64 to 192 was not run |
| Throughput, relaxed TTFT (seconds are fine) | vLLM TP8+EP8, FP8 KV | 256 sessions: 16.08 req/s, 402 tok/s per GPU, TTFT p90 4.1 s | FP8 KV accuracy; ITL p90 56 ms |
| Lowest median TTFT at moderate load | P/D 1P:1D with FP8 KV | 64 sessions: TTFT p50 409 ms (p90 1,022 ms), 99.6% in target, 8% less throughput than aggregated with BF16 KV (topology and KV dtype both differ) | TTFT p90 and p99 are worse than aggregated |
| Smooth streaming under heavy load, TTFT irrelevant | P/D 1P:1D with FP8 KV | 256 sessions: ITL p90 38.5 ms, but TTFT p90 44 s or more (run stopped at its time cap) and 2.4x less throughput | rarely the right trade on one node |
| 1,000 sessions on one node at any target | none measured works | 0% in target on both configurations run | needs more KV (offload) or more GPUs |

## Knob map

<p align="center">
<img src="../figures/f09-knob-map.svg" width="100%" alt="Table of colored cells: for each single change, the baseline and variant values and percent change of TTFT p90, ITL p90, output tok/s per GPU and goodput per GPU. FP8 KV at 256 sessions: TTFT p90 -71%, ITL p90 -66%, output +98%, goodput 2.6x. TP8 to TP8+EP8 at 64: TTFT p90 -30%. DP8+EP8: TTFT p90 3.7x worse. KV router on DP8: TTFT p90 -41%. P/D at 256: TTFT p90 10.8x worse, ITL p90 -32%, output -58%. 64 to 256 sessions on TP8 BF16: TTFT p90 14.8x, ITL p90 8.1x, goodput -100%.">
</p>

> <sub><i>Most knobs trade one latency for another or move results by tens of percent; only more KV (FP8) changes the regime.</i></sub>

| Change | TTFT p90 | ITL p90 | output tok/s/GPU | goodput | Comparison |
|---|---|---|---|---|---|
| BF16 &rarr; FP8 KV (TP8+EP8, 256) | 14.0 &rarr; 4.10 s | 164.6 &rarr; 56.4 ms | 203 &rarr; 402 | 0.14 &rarr; 0.37 | like-for-like |
| TP8 &rarr; TP8+EP8 (64) | 1,214 &rarr; 851 ms | 26.7 &rarr; 25.6 ms | 232 &rarr; 244 | 1.16 &rarr; 1.22 | like-for-like |
| TP8 &rarr; DP8+EP8, vLLM balancer (64) | 1,214 &rarr; 4,534 ms | 26.7 &rarr; 41.4 ms | 232 &rarr; 151 | 1.16 &rarr; 0.32 | like-for-like |
| vLLM balancer &rarr; Dynamo KV router (DP8+EP8, 64) | 4,534 &rarr; 2,687 ms | 41.4 &rarr; 36.6 ms | 151 &rarr; 172 | 0.32 &rarr; 0.55 | like-for-like |
| Default &rarr; tuned MoE config (TP8, S2 16) | 4.78 &rarr; 4.09 s | 54.5 &rarr; 49.1 ms | 32 &rarr; 36 | 0.005 &rarr; 0.093 | like-for-like |
| Default &rarr; tuned MoE config (TP8, 64) | 1,230 &rarr; 2,026 ms | 26.8 &rarr; 26.3 ms | 231 &rarr; 223 | 1.16 &rarr; 0.98 | like-for-like, unexplained |
| vLLM &rarr; TensorRT-LLM (TP8+EP8, 64) | 851 &rarr; 733 ms | 25.6 &rarr; 27.7 ms | 244 &rarr; 225 | 1.22 &rarr; 1.12 | KV pool -5% |
| Aggregated &rarr; P/D, BF16 KV (64) | 851 &rarr; 3,251 ms | 25.6 &rarr; 27.1 ms | 244 &rarr; 206 | 1.22 &rarr; 0.89 | like-for-like |
| Aggregated &rarr; P/D, FP8 KV (256) | 4.10 &rarr; 44.4 s | 56.4 &rarr; 38.5 ms | 402 &rarr; 167 | 0.37 &rarr; 0.23 | P/D stopped at time cap |
| 8 &rarr; 20 turns per session (TP8, 64) | 1,214 &rarr; 1,249 ms | 26.7 &rarr; 33.1 ms | 232 &rarr; 163 | 1.16 &rarr; 0.82 | completion rules differ |
| 64 &rarr; 256 sessions (TP8, BF16 KV) | 1,214 ms &rarr; 17.9 s | 26.7 &rarr; 216 ms | 232 &rarr; 165 | 1.16 &rarr; 0.0004 | like-for-like |

Not measured yet: chunked-prefill budget (`max_num_batched_tokens`), CUDA graph sizes, attention backend, MoE backend override, speculative decoding, host-memory KV offload, TensorRT-LLM tuning, other P:D ratios ([docs/08](08-limits-and-next-measurements.md)).

## Sizing

- **Plan around 64 concurrent agentic sessions per 8x H100 node** for this model at this target, on BF16 KV. That is the highest measured load that meets the target (here every request was in target); the true limit lies between 64 and 256 and may be higher with FP8 KV once its accuracy is confirmed.
- **Node count follows from it**: 1,000 concurrent sessions at this target need about 16 nodes at the measured 64 per node (1,000 / 64 = 15.6). Measuring 96 to 192 sessions could lower that.
- **Longer sessions lower the number.** 20-turn sessions at 64 already use 85% of the BF16 pool (37% for 8 turns). Recompute session demand (prompt + output tokens at the last turn, times sessions) against the KV pool in [docs/01](01-layouts-and-kv-capacity.md) before reusing 64 for a different agent.
- **A single node cannot hold 1,000 agent sessions of this size**: their end-of-session KV is 6.4M to 8.4M tokens against a 1.81M FP8 pool.

## What to do with it

- **Pick your target first, then the point.** For interactive agents run the 64-session configuration; for batch agents run the FP8 KV configuration after an accuracy check.
- **Before committing a fleet size, measure 96, 128 and 192 sessions with FP8 KV.** That single sweep is the largest expected change to this frontier.
- **Report goodput, not peak throughput.** The highest-throughput point serves 3.3x less goodput per GPU (0.369 vs 1.218 req/s) than the interactive point.

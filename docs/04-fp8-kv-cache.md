# 04 FP8 KV cache

**What you get from this page:** what storing the KV cache in FP8 buys for agentic serving on this node (capacity, throughput, latency), at 256 and at 1,000 sessions, what limit it leaves in place, and the accuracy check it still needs.

## The answer

**An FP8 KV cache is the largest lever measured.** On vLLM TP8+EP8 it doubles the KV pool (906,304 to 1,812,224 tokens) and, at 256 agentic sessions with every other flag equal, doubles throughput (203 to 402 output tok/s per GPU), cuts TTFT p90 by 71% (14.0 to 4.1 s) and ITL p90 by 66% (164.6 to 56.4 ms), and brings the prefix-cache hit back to its ceiling (57.0% to 89.0%). It does **not** make 256 sessions meet the target: only 18% of requests do, because ITL p90 (56.4 ms) and TTFT p90 (4.1 s) still exceed the bounds once KV is no longer the limit. At 1,000 sessions the doubled pool fills again and no request meets the target. FP8 KV ran with an **uncalibrated scale (1.0)**, so output quality is not verified.

**What a good number is:** the target is TTFT &le; 2 s (the MLPerf conversational limit value) and mean ITL &le; 50 ms for at least 90% of requests ([docs/00](00-method.md#the-target-slo-and-why-these-bounds)). At 256 sessions FP8 KV gets 18% of requests there, BF16 KV 14%; at 64 sessions (BF16 KV) 100% are.

## Configuration

The TP8+EP8 command plus `--kv-cache-dtype fp8`, nothing else ([data/configs/vllm-tp8ep8-fp8kv/cmd.txt](../data/configs/vllm-tp8ep8-fp8kv/cmd.txt)). The startup log reports 1,812,224 tokens, exactly 2.0x the BF16 pool on the same layout, and warns: `Using KV cache scaling factor 1.0 for fp8_e4m3`. The checkpoint carries no KV scales, and no accuracy test was run with FP8 KV.

## At 256 sessions: like-for-like

Same layout, same flags apart from the KV dtype, same 2,208 turns, both runs finished every planned turn. Cells show **p90** (p50 / p99).

| Metric | BF16 KV | FP8 KV | Change |
|---|---|---|---|
| TTFT | **14.0 s** (3.34 / 28.4) | **4.10 s** (2.06 / 5.31) | p90 -71%, p99 -81% |
| ITL | **164.6 ms** (57.2 / 166.5) | **56.4 ms** (45.1 / 61.1) | p90 -66% |
| tok/s per user (p90) | 6.1 | 17.7 | 2.9x |
| req/s | 8.10 | 16.08 | 1.98x |
| output tok/s per GPU | 203 | 402 | 1.98x |
| in target / goodput per GPU | 14% / 0.140 | 18% / 0.369 | goodput 2.64x |
| prefix hit (ceiling 89.7% for these sessions) | 57.0% | 89.0% | back to the ceiling |
| KV peak / preemptions / waiting p90 | 1.00 / 20 / 95 | 0.70 / 0 / 106 | |
| end-to-end latency p90 | 46.7 s | 12.3 s | |

Sources: `data/vllm-tp8ep8/AG-c256-t8`, `data/vllm-tp8ep8-fp8kv/AG-c256-t8`.

<p align="center">
<img src="../figures/anim/a05-session-growth.svg" width="100%" alt="Animation: per-turn TTFT p90 and ITL p90 at 256 sessions on TP8+EP8. BF16 KV jumps at turns 4 to 5 (TTFT p90 to 28.4 s, ITL p90 above 150 ms); FP8 KV stays between 2.1 and 5.1 s TTFT p90 and 43 to 59 ms ITL p90 on every turn.">
</p>

> <sub><i>FP8 KV removes the mid-session cliff at 256 sessions; what remains is a flat latency that sits just above the target.</i></sub>

What is left after the fix: KV peak is 0.70 with no preemptions, yet 106 requests wait at the p90 and ITL p90 is 56 ms. The limit has moved from KV capacity to step time: each engine step mixes the running decodes with up to 8,192 prefill tokens (vLLM's default chunked-prefill budget). Lowering that budget would trade TTFT for ITL; it was not measured here.

## At 1,000 sessions: not enough

| Metric | TP8, BF16 KV | TP8+EP8, FP8 KV |
|---|---|---|
| Turns completed of 8,550 planned (time cap 360 s + 120 s) | 4,122 | 5,246 |
| TTFT | **99.3 s** (50.9 / 117.1) | **72.5 s** (21.0 / 83.3) |
| ITL | **265 ms** (230 / 269) | **278 ms** (257 / 280) |
| req/s / output tok/s per GPU | 8.59 / 215 | 11.64 / 291 |
| in target | 0% | 0% |
| prefix hit / KV peak | 50.8% / 1.00 | 51.5% / 1.00 |
| preemptions / waiting p90 | 108 / 701 | 9 / 549 |

Sources: `data/vllm-tp8-matrix/AG-c1000-t8`, `data/vllm-tp8ep8-fp8kv/AG-c1000-t8`. Both runs stopped while latency was still rising, so these are lower bounds. The rows differ in layout **and** KV dtype (no TP8+EP8 BF16 run at 1,000 exists), so the difference is not an FP8 effect size. 1,000 sessions at 5k to 8k tokens each need several million tokens of KV, at least 3x the FP8 pool.

## What to do with it

- **Turn on FP8 KV for agentic load on this model only after an accuracy check** (for example GSM8K or MMLU-Pro parity against BF16 KV, or calibrated KV scales). The performance case is strong; the quality case is untested.
- **Expect it to move capacity, not to meet a strict target by itself.** At 256 sessions it doubles throughput but leaves ITL p90 just above 50 ms; pair it with a smaller chunked-prefill budget or fewer sessions if ITL is binding.
- **Measure FP8 KV at 96 to 192 sessions** before sizing a fleet: that range was not run, and it is where the best operating point that still meets the target most likely sits ([docs/06](06-operating-frontier-and-sizing.md)).
- **For 1,000 sessions per node**, KV dtype alone is not enough; the options are host-memory KV offload, more GPUs, or shorter retained context, none of which was measured here.

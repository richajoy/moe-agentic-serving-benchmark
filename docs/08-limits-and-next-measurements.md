# 08 Limits and next measurements

**What you get from this page:** everything this benchmark does not show, how much each gap could change its answers, and the specific runs that would close them, in order of expected impact.

## The answer

The answers on the other pages are directionally solid (the KV cliff between 64 and 256 sessions, FP8 KV as the 2x lever, TP8+EP8 as the layout, P/D losing on one node), but three gaps limit how far the exact numbers can be pushed: **96 to 192 sessions were never measured**, **FP8 KV output quality was never checked**, and **every point is a single run**. Closing the first two would change the recommended operating point; closing the third would change error bars more than rankings.

## What is not shown, ranked by expected effect

| # | Gap | Why it matters | Run that closes it |
|---|---|---|---|
| 1 | No point between 64 and 256 sessions on any configuration | The frontier has two points; the best operating point that still meets the target almost certainly lies in this gap | 96, 128, 160 and 192 sessions on TP8+EP8 with BF16 and FP8 KV |
| 2 | FP8 KV accuracy not checked (uncalibrated scale 1.0) | If FP8 KV degrades output, every FP8 result (402 tok/s per GPU, the 256-session fix) is off the table | GSM8K or MMLU-Pro parity, BF16 KV vs FP8 KV, same server settings; or calibrated KV scales |
| 3 | FP8 KV not run at 64 sessions | Whether FP8 also helps at the strict-target end is unknown (BF16 KV peak there is only 0.37, so likely little) | FP8 KV at 64 sessions, same rule as the BF16 point |
| 4 | No chunked-prefill budget sweep | At FP8 KV and 256 sessions the limit is step time (ITL p90 56.4 ms with KV at 0.70); a smaller budget trades TTFT for ITL and could move that point into target | `max_num_batched_tokens` 2,048 and 4,096 at 256 sessions |
| 5 | No speculative decoding, no agentic point below 64 sessions | The 50 tok/s per user tier is empty for the 8-turn shape | EAGLE-style drafting at 16 to 64 sessions |
| 6 | Single runs, mixed completion rules, no cache reset between points | Differences under about 10% are not distinguishable; time-capped points are lower bounds | one completion rule, a server restart per point, three repeats |
| 7 | Host-memory KV offload not measured | The only path toward 1,000 sessions per node; attempts with Dynamo's KV block manager did not produce a valid point | KV offload to CPU memory at 512 and 1,000 sessions, with the host tier larger than the GPU pool |
| 8 | Other layouts and engines untuned | TensorRT-LLM ran at defaults (MoE backend, attention DP, FP8 KV untried); only one P:D ratio fits one node | TensorRT-LLM MoE backend and FP8 KV; multi-node P/D |
| 9 | Synthetic workload | Real agents have longer, more varied contexts and less cross-session sharing than a 2,000-token prefix common to all sessions | an AgentX-style trace replay ([docs/07](07-how-this-compares.md)) |
| 10 | No kernel profile | The TP8 vs TP8+EP8 gain mixes kernel choice and communication | an Nsight Systems capture of both layouts on the S2 point |

## Caveats that travel with specific numbers

- **64-session points** are count-capped at 512 of 572 planned turns (89.5%): the longest sessions' last turns are missing.
- **Complete 256-session points** include a cheap drain at the end (turns 9 to 13 with TTFT p90 under 1.4 s on FP8 KV), which pulls percentiles down compared with a steady state.
- **512- and 1,000-session points** and **P/D at 256** stopped at their time cap; their latencies are lower bounds. The two 1,000-session runs also differ in layout and KV dtype.
- **TensorRT-LLM** counts prefix hits in blocks and does not export preemptions or queue length; its KV pool is 5% smaller.
- **S1 TTFT on the TP8 run** is not valid (prompts repeated from a small synthetic pool).
- **The tuned MoE config's 64-session regression** (TTFT p90 +65%) is from one run and unexplained.
- **Per-user speed from ITL excludes the wait for the first token.** Including it, the 64-session median request delivers 200 tokens in 5.0 s (40 tok/s); at 256 sessions with FP8 KV, 200 tokens take 11.5 s (17 tok/s).

## What to do with it

- **Before sizing a fleet from this benchmark**, run gap 1 with FP8 KV and gap 2; together they decide whether the plan is 64 sessions per node or something closer to 128 to 192.
- **Before quoting a difference under 10%**, repeat it.
- **Before quoting any 1,000-session latency**, say it is a lower bound from a run that stopped at turn 4 or 5.

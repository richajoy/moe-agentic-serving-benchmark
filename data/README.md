# Data

**What you get from this page:** what each data file contains, how every column is defined, and how to recompute any number in the docs from these files alone.

## The answer

All 37 measured points are here, one directory per point, plus [points.json](points.json) with one row per point. Every number in the docs and figures is read from these files; `python3 scripts/analysis/check_data.py` recomputes SLO-ok and goodput from the per-request records and checks every percentile against the AIPerf summaries (0 mismatches). Conditions for every file: Qwen3-235B-A22B-Instruct-2507 FP8 on one 8x H100 SXM5 node, NVIDIA Dynamo 1.6.0, vLLM 0.31.0 or TensorRT-LLM 1.3.0rc29, AIPerf 0.13.0, single run per point, 8 GPUs per run.

## Layout

```
data/
  points.json                  one row per point (37 rows)
  engine-startup.json          per run and worker: KV pool, weights, MoE kernel, start-up time, from the worker logs
  nixl.json                    per-transfer KV hand-off cost for the four P/D points
  accuracy.json                GSM8K accuracy for the two TP8 MoE-config runs
  configs/<run>/               exact launch command (cmd.txt, launch.sh) and the TensorRT-LLM engine yaml
  <run>/<point>/
    summary.json               AIPerf aggregate metrics for the point
    server_metrics_summary.json  engine metrics scraped during the point (prefix cache, KV usage, preemptions, queue, NIXL)
    records.jsonl.gz           one line per request
  vllm-tp8-moe-{default,tuned}/gsm8k.json   per-question GSM8K results
```

Paths in `configs/*/cmd.txt` use placeholders: `${MODEL_DIR}`, `${RUNS_DIR}`, `${CONFIG_DIR}`, `${MOE_CONFIG_DIR}`.

## Runs

| Run | Layout | Engine | KV dtype | Router | Points |
|---|---|---|---|---|---|
| `vllm-tp8` | TP8 | vLLM | BF16 | round-robin | agentic 16 (4-turn), 64, 256 |
| `vllm-tp8-matrix` | TP8 (second launch) | vLLM | BF16 | round-robin | S1, S2, S3; agentic 512, 1,000; 64 with 20 turns |
| `vllm-tp8ep8` | TP8+EP8 | vLLM | BF16 | round-robin | S2-c16, agentic 64 and 256, S1-c64 |
| `trtllm-tp8ep8` | TP8+EP8 | TensorRT-LLM | BF16 | round-robin | same four points |
| `vllm-tp8ep8-fp8kv` | TP8+EP8 | vLLM | FP8 | round-robin | agentic 256, 1,000 |
| `vllm-dp8ep8` | DP8+EP8 | vLLM | BF16 | vLLM internal balancer | agentic 64, 256 |
| `vllm-dp8ep8-kvrouter` | DP8+EP8 | vLLM | BF16 | Dynamo KV router | agentic 64, 256 |
| `vllm-pd-1p1d` | P/D, TP4+EP4 each | vLLM + NIXL | BF16 | round-robin | agentic 64 |
| `vllm-pd-1p1d-fp8kv` | P/D, TP4+EP4 each | vLLM + NIXL | FP8 | round-robin | agentic 64, 256; S2-c16 |
| `vllm-tp8-moe-default` | TP8 | vLLM | BF16 | round-robin | S2-c16, agentic 64, 256 (vLLM's default MoE config) |
| `vllm-tp8-moe-tuned` | TP8 | vLLM | BF16 | round-robin | same points with the tuned MoE config in `deploy/moe-configs/` |

Point names: `AG-c<sessions>-t<mean turns>` for agentic points; `S1`, `S2`, `S3` plus `-c<concurrency>` for fixed shapes.

## points.json columns

| Column | Meaning |
|---|---|
| `run`, `point`, `source` | directory names and the path of the point's files |
| `workload`, `layout`, `engine`, `kv_dtype`, `router`, `moe_config` | parsed from the launch command and worker log |
| `concurrency`, `sessions`, `turns`, `isl`, `osl`, `shared_system_prompt` | load settings (`isl` is new input per turn; the full prompt grows each turn, see `records.jsonl.gz` `isl`) |
| `completion_rule` | `request-count` or `duration NNNs+GGGs grace` |
| `requests_target`, `records`, `turns_planned_total`, `dataset_coverage_pct`, `ran_to_completion`, `max_turn_index_observed` | how far the point got: requests recorded vs turns in the generated dataset |
| `benchmark_duration_s` | AIPerf profiling time |
| `req_s`, `out_tok_s_gpu` | completed requests per second; output tokens per second / 8 |
| `ttft_p50_ms`, `ttft_p90_ms`, `ttft_p99_ms` | time to first token percentiles |
| `itl_p50_ms`, `itl_p90_ms`, `itl_p99_ms` | per-request mean inter-token latency, percentiles across requests |
| `tok_s_user_p90`, `tok_s_user_p50` | 1000 / ITL p90 (the headline per-user speed, slow tail) and 1000 / ITL p50 |
| `e2e_p50_s`, `e2e_p90_s`, `e2e_p99_s` | end-to-end request latency percentiles |
| `slo_ok_pct`, `goodput_req_s_gpu`, `records_with_ttft` | share of requests with TTFT &le; 2,000 ms and mean ITL &le; 50 ms (a point meets the target when this is &ge; 90); req/s x that share / 8; the denominator |
| `hit_pct`, `hit_unit` | prefix-cache hit over the point, in tokens (vLLM, summed over both P/D pools) or 64-token blocks (TensorRT-LLM); null when nothing was reused |
| `kv_max`, `preemptions`, `wait_p90` | peak KV usage; vLLM preemptions; p90 of the waiting-queue gauge (TensorRT-LLM does not export the last two) |
| `errors` | failed requests (0 on every point) |
| `per_turn_ttft` | agentic points: per 0-based turn index, request count and TTFT p50 / p90 (nearest-rank percentile) |

## Other files

- **summary.json**: the AIPerf `profile_export_aiperf.json` for the point, reduced to workload settings (endpoint URLs, tokenizer path and artifact paths removed). Each metric is `{unit, avg, p1 ... p99, min, max, std}`.
- **server_metrics_summary.json**: Prometheus metrics AIPerf scraped during the profiling phase. `source` names the process: `frontend` (port 8000), `worker0` (8081), `worker1` (8082); in P/D runs `worker0` is the decode worker and `worker1` the prefill worker.
- **records.jsonl.gz**: one object per request with `conversation_id`, `turn_index`, `session_num`, `ttft_ms`, `itl_ms`, `request_latency_ms`, `isl` (full prompt tokens including history) and `osl`. Prompts and request ids are not included.
- **nixl.json**: per P/D point, decode-side means per transfer: `transfers`, `mb_per_transfer`, `xfer_ms`, `post_ms`, `post_share_pct`, `descriptors_per_transfer`, `gb_s_per_transfer`, and failure counters (all 0). One transfer is one request's KV for one decode GPU.
- **engine-startup.json**: per run and worker: `kv_tokens`, `kv_tokens_per_rank`, `available_kv_gib_per_gpu`, `weights_gib_per_gpu`, `moe_backend`, `attention_backend`, `triton_moe_config`, `moe_block_refinement`, `engine_init_s`, `compile_s`, `log_start_to_endpoint_registered_s`.
- **accuracy.json** and **gsm8k.json**: GSM8K on the first 400 test questions, greedy decoding through the chat endpoint: 378 / 400 (default MoE config) and 379 / 400 (tuned). Per question: index, extracted answer, reference answer, match.

## How to recompute

```bash
python3 scripts/analysis/check_data.py              # every point, p90-first table, 0 mismatches expected
python3 scripts/analysis/check_data.py --frontier   # plus the operating frontier
```

or by hand for one point:

```python
import gzip, json
recs = [json.loads(l) for l in gzip.open("data/vllm-tp8ep8/AG-c64-t8/records.jsonl.gz", "rt")]
ok = sum(r["ttft_ms"] <= 2000 and r["itl_ms"] <= 50 for r in recs)
s = json.load(open("data/vllm-tp8ep8/AG-c64-t8/summary.json"))
print("SLO-ok %:", 100 * ok / len(recs), "goodput/GPU:", s["request_throughput"]["avg"] * ok / len(recs) / 8)
print("TTFT p90 ms:", s["time_to_first_token"]["p90"], "ITL p90 ms:", s["inter_token_latency"]["p90"],
      "tok/s per user (p90):", 1000 / s["inter_token_latency"]["p90"])
```

Percentiles in `summary.json` are AIPerf's own; recomputing them from `records.jsonl.gz` with a different percentile method can differ in the last digit.

## What to do with it

Cite a number with its `run/point` path and completion rule. Compare two rows only after checking `completion_rule` and `dataset_coverage_pct`: count-capped, complete and stopped-early points are not interchangeable ([docs/00](../docs/00-method.md#completion-rules)).

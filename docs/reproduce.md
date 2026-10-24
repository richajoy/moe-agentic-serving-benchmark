# Reproduce

Back to the [overview](../README.md).

## Re-run the benchmark

You need one 8x H100 (80 GB, NVLink) node, Docker with the NVIDIA container toolkit, the FP8 checkpoint, and AIPerf 0.13 on the host.

```bash
export MODEL_DIR=/path/to/models RUNS_DIR=/path/to/runs CACHE_DIR=/path/to/cache CONFIG_DIR=$PWD/deploy
# 1. serve: Dynamo frontend + one vLLM TP8+EP8 worker with an FP8 KV cache (the 256-session throughput point)
scripts/serve/up.sh vllm-tp8ep8-fp8kv round-robin \
  "PYTHONHASHSEED=0|--model $MODEL_DIR/Qwen3-235B-A22B-Instruct-2507-FP8 --served-model-name qwen235 --max-model-len 40960 \
   --gpu-memory-utilization 0.92 --block-size 64 --max-num-seqs 1024 --enable-prefix-caching \
   --tensor-parallel-size 8 --enable-expert-parallel --kv-cache-dtype fp8"
scripts/serve/wait-ready.sh
# 2. load: one agentic point (AG = 8-turn sessions), 256 concurrent sessions, 300 s cap + 120 s grace
DUR=300 GRACE=120 scripts/bench/bench.sh vllm-tp8ep8-fp8kv AG 256 8
# 3. summarize: one row per point, same math as data/points.json
python3 scripts/analysis/summarize_aiperf.py $RUNS_DIR/vllm-tp8ep8-fp8kv/AG-c256-t8
```

- The exact launch command of every run is in [data/configs/](../data/configs/); the point sequences are in [scripts/bench/](../scripts/bench/); TensorRT-LLM and P/D launches are in [scripts/serve/](../scripts/serve/) and [deploy/](../deploy/).
- Regenerate every figure from the committed data: `python3 scripts/figures/make_figures.py && python3 scripts/figures/make_animations.py` (Python 3, no dependencies).
- **Known issue:** vLLM 0.31.0, the version in the Dynamo 1.6.0 vLLM image, can hang at startup on a warm restart of an expert-parallel or data-parallel MoE worker, inside FlashInfer's MoE autotune step ([vllm#57423](https://github.com/vllm-project/vllm/issues/57423), fixed in vLLM 0.32.0 by [vllm#57635](https://github.com/vllm-project/vllm/pull/57635)). Use vLLM 0.32.0 or later, or clear the FlashInfer autotune cache before each launch, as `scripts/serve/up.sh` does.

## Limits

- **Single run per point**, no repeats. Treat differences under about 10% as noise. The prefix cache was not reset between points on the same server.
- **Mixed completion rules.** 64-session points stop after a fixed request count (89.5% of planned turns); some 256-session points finished every turn; 512- and 1,000-session points stopped at a time cap, so their latencies are lower bounds. Each table says which.
- **FP8 KV is uncalibrated** (scale 1.0) and output quality was not checked, so every FP8 KV result is a performance result only.
- **96 to 192 sessions were not measured**, and FP8 KV was not run at 64 sessions, so the highest load that meets the target is unknown; 64 is the highest measured.
- **Synthetic workload**: forced output lengths, synthetic prompts, a 2,000-token prompt shared by every session. Real agent traffic has longer contexts and different reuse.
- **One node, default engine knobs**: no chunked-prefill budget sweep, no speculative decoding, no host-memory KV offload, one P/D ratio. Full list: [08 Limits and next measurements](08-limits-and-next-measurements.md).

## Repository layout

| Path | Contents |
|---|---|
| `docs/` | Method, six result pages, comparisons, limits, glossary |
| `data/` | Per-point AIPerf summaries, scraped engine metrics and per-request records for all 37 points; `points.json` holds one row per point |
| `figures/` | Static figures (`f0*.svg`) and animations (`anim/a0*.svg`), all generated from `data/` |
| `scripts/serve/` | Launch the Dynamo frontend and vLLM or TensorRT-LLM workers in one container; readiness check |
| `scripts/bench/` | One AIPerf point (`bench.sh`) and the point sequences used for each run |
| `scripts/analysis/` | Summarize raw AIPerf output; verify every published number from `data/`; NCCL bandwidth check |
| `scripts/figures/` | Figure and animation generators (stdlib Python) |
| `deploy/` | TensorRT-LLM engine configs and the tuned fused-MoE config for pure TP8 |

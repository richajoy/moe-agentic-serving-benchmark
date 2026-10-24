# Tuned fused-MoE config for Qwen3-235B-A22B FP8 on pure TP8 (H100)

On pure TP8, each GPU holds a 192-wide slice of every expert (1536 / 8), so vLLM refines the FP8 scale blocks to [64, 64] and runs its Triton MoE kernel. vLLM 0.31.0 ships no tuned Triton config for that shape on H100 and logs `Using default MoE config`.

The JSON file in this directory is a tuned Triton config for exactly that shape: `E=128, N=192, dtype=fp8_w8a8, block_shape=[64,64]`, H100 80GB HBM3, Triton 3.7.1. Use it with:

```bash
export VLLM_TUNED_CONFIG_FOLDER=/path/to/deploy/moe-configs   # passed to the worker environment
```

The worker log then shows `Using configuration from .../E=128,N=192,...,block_shape=[64,64].json for MoE layer`.

Measured effect (pure TP8, BF16 KV, single paired runs, [docs/01](../../docs/01-layouts-and-kv-capacity.md#a-tuned-moe-config-for-pure-tp8)): +11.5% req/s on the prefill-heavy S2 shape and +8.2% on agentic 256 sessions (both runs stopped at a 240 s time cap, so a capped-window rate), but -3.6% at agentic 64 sessions with a worse TTFT tail (TTFT p90 +65%, unexplained, one run); GSM8K accuracy unchanged (378 vs 379 of 400). TP8 attention + EP8 experts avoids this kernel path entirely and stays faster, so prefer that layout; this config matters only if you must run pure TP8.

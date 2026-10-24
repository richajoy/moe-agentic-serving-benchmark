#!/bin/bash
# Points on vLLM TP8 (runs vllm-tp8 and vllm-tp8-matrix): agentic 16 (4 turns), 64 and 256 sessions (count-capped),
# fixed shapes S1/S2/S3, agentic 512 and 1,000 sessions (360 s cap + 120 s grace), 20-turn sessions at 64.
# The published data used two launches of the same TP8 command (runs vllm-tp8 and vllm-tp8-matrix).
B=$(dirname "$0"); export TMO=1200
"$B/../serve/wait-ready.sh" || exit 1
"$B/bench.sh" vllm-tp8 AG 16 4; "$B/bench.sh" vllm-tp8 AG 64 8; "$B/bench.sh" vllm-tp8 AG 256 8
for c in 16 64 256; do "$B/bench.sh" vllm-tp8-matrix S2 $c; done
for c in 16 64 256; do "$B/bench.sh" vllm-tp8-matrix S1 $c; done
"$B/bench.sh" vllm-tp8-matrix S3 16
for c in 512 1000; do DUR=360 GRACE=120 "$B/bench.sh" vllm-tp8-matrix AG $c 8; done
DUR=300 GRACE=120 "$B/bench.sh" vllm-tp8-matrix AG 64 20

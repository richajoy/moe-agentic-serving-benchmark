#!/bin/bash
# run-engine-points.sh RUN: the four points used to compare vLLM and TensorRT-LLM at TP8+EP8, in this order.
B=$(dirname "$0"); R=$1; export TMO=1200
"$B/../serve/wait-ready.sh" || exit 1
"$B/bench.sh" $R S2 16; "$B/bench.sh" $R AG 64 8; "$B/bench.sh" $R S1 64; DUR=300 GRACE=120 "$B/bench.sh" $R AG 256 8

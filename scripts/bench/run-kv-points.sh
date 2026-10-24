#!/bin/bash
# run-kv-points.sh RUN: agentic 256 sessions (300 s cap + 120 s grace) and 1,000 sessions (360 s + 120 s), for KV-capacity arms.
B=$(dirname "$0"); R=$1; export TMO=1200
"$B/../serve/wait-ready.sh" || exit 1
DUR=300 GRACE=120 "$B/bench.sh" $R AG 256 8
DUR=360 GRACE=120 "$B/bench.sh" $R AG 1000 8

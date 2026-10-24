#!/bin/bash
# run-pd-points.sh RUN: P/D points. AIPerf scrapes both workers (8081 decode, 8082 prefill).
B=$(dirname "$0"); R=$1; export TMO=1200; export METRICS_URLS="http://localhost:8081/metrics http://localhost:8082/metrics"
"$B/../serve/wait-ready.sh" || exit 1
"$B/bench.sh" $R AG 64 8
DUR=300 GRACE=120 "$B/bench.sh" $R AG 256 8
"$B/bench.sh" $R S2 16

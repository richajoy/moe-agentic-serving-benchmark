#!/bin/bash
python -m dynamo.frontend --http-port 8000 --router-mode kv --kv-cache-block-size 64 > ${RUNS_DIR}/vllm-dp8ep8-kvrouter/frontend.log 2>&1 &
env DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 PYTHONHASHSEED=0 python -m dynamo.vllm --model ${MODEL_DIR}/Qwen3-235B-A22B-Instruct-2507-FP8 --served-model-name qwen235 --max-model-len 40960 --gpu-memory-utilization 0.92 --block-size 64 --enable-prefix-caching --max-num-seqs 256 --data-parallel-size 8 --enable-expert-parallel --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' > ${RUNS_DIR}/vllm-dp8ep8-kvrouter/worker0.log 2>&1 &
wait
